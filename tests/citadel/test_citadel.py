from sqlalchemy.orm import Session

from acn.citadel import (
    AuditDecision,
    CitadelActionRequest,
    CitadelSafetyLayer,
    OverrideApproval,
    SqlAlchemyAuditLogRepository,
)
from acn.controller import AdaptiveAction
from acn.versioning.repository import SqlAlchemyTrainingVersionRepository


def _version_repository(session: Session) -> SqlAlchemyTrainingVersionRepository:
    repository = SqlAlchemyTrainingVersionRepository(session)
    first_checkpoint = repository.create_checkpoint(uri="s3://mlflow/a.pt", content_hash="sha:a")
    second_checkpoint = repository.create_checkpoint(uri="s3://mlflow/b.pt", content_hash="sha:b")
    repository.create_branch(name="main")
    repository.create_commit(
        branch_name="main",
        checkpoint_id=first_checkpoint.id,
        message="first",
        commit_id="cmt_first",
    )
    repository.create_commit(
        branch_name="main",
        checkpoint_id=second_checkpoint.id,
        message="second",
        commit_id="cmt_second",
    )
    return repository


def test_allows_safe_rollback_and_records_audit(session: Session) -> None:
    version_repository = _version_repository(session)
    audit_repository = SqlAlchemyAuditLogRepository(session)
    citadel = CitadelSafetyLayer(
        version_repository=version_repository,
        audit_repository=audit_repository,
    )

    result = citadel.validate_action(
        CitadelActionRequest(
            action=AdaptiveAction.ROLLBACK,
            actor="operator",
            branch_name="main",
            current_commit_id="cmt_second",
            parameters={"target_commit_id": "cmt_first"},
        )
    )

    assert result.allowed
    assert result.decision is AuditDecision.ALLOWED
    assert audit_repository.list_records()[0].action == AdaptiveAction.ROLLBACK.value


def test_denies_unsafe_rollback_target(session: Session) -> None:
    version_repository = _version_repository(session)
    audit_repository = SqlAlchemyAuditLogRepository(session)
    citadel = CitadelSafetyLayer(
        version_repository=version_repository,
        audit_repository=audit_repository,
    )

    result = citadel.validate_action(
        CitadelActionRequest(
            action=AdaptiveAction.ROLLBACK,
            actor="operator",
            branch_name="main",
            current_commit_id="cmt_second",
            parameters={"target_commit_id": "cmt_unknown"},
        )
    )

    assert not result.allowed
    assert result.decision is AuditDecision.DENIED
    assert not result.requires_override
    assert "not reachable" in result.reasons[0]


def test_validates_learning_rate_bounds() -> None:
    citadel = CitadelSafetyLayer()

    result = citadel.validate_action(
        CitadelActionRequest(
            action=AdaptiveAction.INCREASE_LEARNING_RATE,
            actor="operator",
            branch_name="main",
            parameters={"learning_rate": 2.0},
        )
    )

    assert not result.allowed
    assert result.requires_override
    assert "exceeds" in result.reasons[0]


def test_override_can_approve_learning_rate_policy_violation() -> None:
    citadel = CitadelSafetyLayer()

    result = citadel.validate_action(
        CitadelActionRequest(
            action=AdaptiveAction.INCREASE_LEARNING_RATE,
            actor="operator",
            branch_name="main",
            parameters={"learning_rate": 2.0},
            override=OverrideApproval(
                approved_by="lead",
                reason="Controlled recovery experiment.",
                ticket_id="ACN-42",
            ),
        )
    )

    assert result.allowed
    assert result.decision is AuditDecision.OVERRIDE_APPROVED
    record = citadel.audit_repository.list_records()[0]
    assert record.override_by == "lead"
    assert record.override_ticket_id == "ACN-42"


def test_denies_stable_checkpoint_overwrite_even_without_override() -> None:
    citadel = CitadelSafetyLayer()

    result = citadel.validate_checkpoint_registration(
        actor="worker",
        branch_name="main",
        uri="s3://mlflow/stable.pt",
        content_hash="sha:stable",
        overwrite_existing=True,
    )

    assert not result.allowed
    assert result.decision is AuditDecision.DENIED
    assert "immutable" in result.reasons[0]


def test_allows_valid_stable_checkpoint_registration() -> None:
    citadel = CitadelSafetyLayer()

    result = citadel.validate_checkpoint_registration(
        actor="worker",
        branch_name="main",
        uri="s3://mlflow/stable.pt",
        content_hash="sha:stable",
        overwrite_existing=False,
    )

    assert result.allowed
    assert result.decision is AuditDecision.ALLOWED


def test_denies_layer_action_without_selector() -> None:
    citadel = CitadelSafetyLayer()

    result = citadel.validate_action(
        CitadelActionRequest(
            action=AdaptiveAction.FREEZE_LAYERS,
            actor="controller",
            branch_name="main",
        )
    )

    assert not result.allowed
    assert result.requires_override
    assert "layer_selector" in result.reasons[0]


def test_denies_experimental_branch_from_unreachable_commit(session: Session) -> None:
    version_repository = _version_repository(session)
    citadel = CitadelSafetyLayer(version_repository=version_repository)

    result = citadel.validate_action(
        CitadelActionRequest(
            action=AdaptiveAction.CREATE_EXPERIMENTAL_BRANCH,
            actor="controller",
            branch_name="main",
            parameters={"source_commit_id": "cmt_missing"},
        )
    )

    assert not result.allowed
    assert result.requires_override
    assert "not reachable" in result.reasons[0]
