from dataclasses import dataclass, field

from acn.citadel import CitadelActionRequest, CitadelSafetyLayer
from acn.controller import AdaptiveAction, ControllerDecision
from acn.orchestration.rollback import RollbackCoordinator
from acn.versioning.domain import BranchRecord, Metadata
from acn.versioning.repository import TrainingVersionRepository


@dataclass(frozen=True, slots=True)
class DecisionExecutionResult:
    action: AdaptiveAction
    executed: bool
    message: str
    metadata: Metadata = field(default_factory=dict)


class DecisionExecutor:
    def __init__(
        self,
        *,
        version_repository: TrainingVersionRepository,
        citadel: CitadelSafetyLayer,
        rollback_coordinator: RollbackCoordinator,
    ) -> None:
        self._version_repository = version_repository
        self._citadel = citadel
        self._rollback_coordinator = rollback_coordinator

    def execute(
        self,
        *,
        decision: ControllerDecision,
        actor: str,
        branch_name: str,
        current_commit_id: str | None,
    ) -> DecisionExecutionResult:
        if decision.action is AdaptiveAction.CONTINUE_TRAINING:
            return DecisionExecutionResult(
                action=decision.action,
                executed=True,
                message="Training continues without orchestration mutation.",
            )
        if decision.action is AdaptiveAction.ROLLBACK:
            target_commit_id = _required_string(decision.parameters.get("target_commit_id"))
            branch = self._rollback_coordinator.rollback(
                actor=actor,
                branch_name=branch_name,
                current_commit_id=current_commit_id,
                target_commit_id=target_commit_id,
            )
            return _branch_result(decision.action, branch, "Rollback executed.")
        if decision.action is AdaptiveAction.CREATE_EXPERIMENTAL_BRANCH:
            return self._create_experimental_branch(
                decision=decision,
                actor=actor,
                branch_name=branch_name,
            )

        validation = self._citadel.validate_action(
            CitadelActionRequest(
                action=decision.action,
                actor=actor,
                branch_name=branch_name,
                current_commit_id=current_commit_id,
                parameters=decision.parameters,
            )
        )
        if not validation.allowed:
            return DecisionExecutionResult(
                action=decision.action,
                executed=False,
                message="Decision denied by Citadel.",
                metadata={"reasons": list(validation.reasons)},
            )

        return DecisionExecutionResult(
            action=decision.action,
            executed=True,
            message="Decision validated for downstream trainer execution.",
            metadata=dict(decision.parameters),
        )

    def _create_experimental_branch(
        self,
        *,
        decision: ControllerDecision,
        actor: str,
        branch_name: str,
    ) -> DecisionExecutionResult:
        validation = self._citadel.validate_action(
            CitadelActionRequest(
                action=decision.action,
                actor=actor,
                branch_name=branch_name,
                parameters=decision.parameters,
            )
        )
        if not validation.allowed:
            return DecisionExecutionResult(
                action=decision.action,
                executed=False,
                message="Experimental branch creation denied by Citadel.",
                metadata={"reasons": list(validation.reasons)},
            )

        source_commit_id = _required_string(decision.parameters.get("source_commit_id"))
        new_branch_name = f"{branch_name}/exp-{source_commit_id[:8]}"
        branch = self._version_repository.create_branch(
            name=new_branch_name,
            base_commit_id=source_commit_id,
            metadata={"created_by": "decision_executor"},
        )
        return _branch_result(decision.action, branch, "Experimental branch created.")


def _required_string(value: object) -> str:
    if isinstance(value, str) and value:
        return value
    msg = "Decision parameter must be a non-empty string."
    raise ValueError(msg)


def _branch_result(
    action: AdaptiveAction,
    branch: BranchRecord,
    message: str,
) -> DecisionExecutionResult:
    return DecisionExecutionResult(
        action=action,
        executed=True,
        message=message,
        metadata={
            "branch_name": branch.name,
            "head_commit_id": branch.head_commit_id,
            "base_commit_id": branch.base_commit_id,
        },
    )
