import logging
from dataclasses import dataclass, field

from acn.citadel.audit import AuditLogRepository, InMemoryAuditLogRepository
from acn.citadel.domain import (
    AuditDecision,
    CitadelActionRequest,
    CitadelValidationResult,
)
from acn.controller.domain import AdaptiveAction, DecisionParameter
from acn.versioning.repository import TrainingVersionRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CitadelPolicyConfig:
    critical_actions: frozenset[AdaptiveAction] = field(
        default_factory=lambda: frozenset(
            {
                AdaptiveAction.ROLLBACK,
                AdaptiveAction.FREEZE_LAYERS,
                AdaptiveAction.UNFREEZE_LAYERS,
                AdaptiveAction.CREATE_EXPERIMENTAL_BRANCH,
                AdaptiveAction.INCREASE_LEARNING_RATE,
                AdaptiveAction.DECREASE_LEARNING_RATE,
            }
        )
    )
    override_allowed_actions: frozenset[AdaptiveAction] = field(
        default_factory=lambda: frozenset(
            {
                AdaptiveAction.DECREASE_LEARNING_RATE,
                AdaptiveAction.INCREASE_LEARNING_RATE,
                AdaptiveAction.FREEZE_LAYERS,
                AdaptiveAction.UNFREEZE_LAYERS,
                AdaptiveAction.CREATE_EXPERIMENTAL_BRANCH,
            }
        )
    )
    minimum_learning_rate: float = 1e-7
    maximum_learning_rate: float = 1.0
    allow_rollback_without_best_commit: bool = False


class CitadelSafetyLayer:
    def __init__(
        self,
        *,
        version_repository: TrainingVersionRepository | None = None,
        audit_repository: AuditLogRepository | None = None,
        config: CitadelPolicyConfig | None = None,
    ) -> None:
        self._version_repository = version_repository
        self._audit_repository = audit_repository or InMemoryAuditLogRepository()
        self._config = config or CitadelPolicyConfig()

    @property
    def audit_repository(self) -> AuditLogRepository:
        return self._audit_repository

    def validate_action(self, request: CitadelActionRequest) -> CitadelValidationResult:
        result = self._validate_action(request)
        self._record_audit(request, result)
        return result

    def validate_checkpoint_registration(
        self,
        *,
        actor: str,
        branch_name: str,
        uri: str,
        content_hash: str,
        overwrite_existing: bool,
    ) -> CitadelValidationResult:
        reasons: list[str] = []
        if overwrite_existing:
            reasons.append(
                "Stable checkpoint overwrite is denied because checkpoints are immutable."
            )
        if not uri:
            reasons.append("Checkpoint URI is required.")
        if not content_hash:
            reasons.append("Checkpoint content hash is required.")

        decision = AuditDecision.DENIED if reasons else AuditDecision.ALLOWED
        result = CitadelValidationResult(
            allowed=not reasons,
            decision=decision,
            reasons=tuple(reasons) if reasons else ("Stable checkpoint registration is safe.",),
        )
        self._audit_repository.record(
            action="checkpoint_registration",
            actor=actor,
            branch_name=branch_name,
            decision=result.decision,
            reasons=result.reasons,
            parameters={
                "uri": uri,
                "content_hash": content_hash,
                "overwrite_existing": overwrite_existing,
            },
        )
        return result

    def _validate_action(self, request: CitadelActionRequest) -> CitadelValidationResult:
        reasons: list[str] = []

        if request.action not in self._config.critical_actions:
            return CitadelValidationResult(
                allowed=True,
                decision=AuditDecision.ALLOWED,
                reasons=("Action is non-critical and allowed by Citadel policy.",),
            )

        if request.action is AdaptiveAction.ROLLBACK:
            reasons.extend(self._validate_rollback(request))
        elif request.action in {
            AdaptiveAction.DECREASE_LEARNING_RATE,
            AdaptiveAction.INCREASE_LEARNING_RATE,
        }:
            reasons.extend(self._validate_learning_rate(request.parameters.get("learning_rate")))
        elif request.action in {AdaptiveAction.FREEZE_LAYERS, AdaptiveAction.UNFREEZE_LAYERS}:
            reasons.extend(self._validate_layer_selector(request.parameters.get("layer_selector")))
        elif request.action is AdaptiveAction.CREATE_EXPERIMENTAL_BRANCH:
            reasons.extend(self._validate_experimental_branch(request))

        if not reasons:
            return CitadelValidationResult(
                allowed=True,
                decision=AuditDecision.ALLOWED,
                reasons=("Critical action passed Citadel validation.",),
            )

        if self._override_is_valid(request):
            return CitadelValidationResult(
                allowed=True,
                decision=AuditDecision.OVERRIDE_APPROVED,
                reasons=(*reasons, "Explicit override approval accepted."),
            )

        return CitadelValidationResult(
            allowed=False,
            decision=AuditDecision.DENIED,
            reasons=tuple(reasons),
            requires_override=request.action in self._config.override_allowed_actions,
        )

    def _validate_rollback(self, request: CitadelActionRequest) -> list[str]:
        target_commit_id = _string_parameter(request.parameters.get("target_commit_id"))
        if target_commit_id is None:
            if self._config.allow_rollback_without_best_commit:
                return []
            return ["Rollback requires a target commit id."]

        if self._version_repository is None:
            return ["Rollback safety requires a version repository."]

        history = self._version_repository.list_branch_history(request.branch_name)
        reachable_commit_ids = {commit.id for commit in history}
        if target_commit_id not in reachable_commit_ids:
            return ["Rollback target is not reachable from the current branch head."]

        if (
            request.current_commit_id is not None
            and request.current_commit_id not in reachable_commit_ids
        ):
            return ["Current commit is not reachable from the current branch head."]

        return []

    def _validate_learning_rate(self, value: DecisionParameter) -> list[str]:
        if not isinstance(value, int | float):
            return ["Learning-rate action requires a numeric learning_rate parameter."]

        learning_rate = float(value)
        if learning_rate < self._config.minimum_learning_rate:
            return ["Requested learning rate is below the Citadel minimum."]
        if learning_rate > self._config.maximum_learning_rate:
            return ["Requested learning rate exceeds the Citadel maximum."]
        return []

    def _validate_layer_selector(self, value: DecisionParameter) -> list[str]:
        if not isinstance(value, str) or not value:
            return ["Layer action requires a non-empty layer_selector parameter."]
        return []

    def _validate_experimental_branch(self, request: CitadelActionRequest) -> list[str]:
        source_commit_id = _string_parameter(request.parameters.get("source_commit_id"))
        if source_commit_id is None:
            return ["Experimental branch creation requires source_commit_id."]
        if self._version_repository is None:
            return []

        history = self._version_repository.list_branch_history(request.branch_name)
        if source_commit_id not in {commit.id for commit in history}:
            return ["Experimental branch source commit is not reachable from branch history."]
        return []

    def _override_is_valid(self, request: CitadelActionRequest) -> bool:
        if request.override is None:
            return False
        if request.action not in self._config.override_allowed_actions:
            return False
        return bool(request.override.approved_by and request.override.reason)

    def _record_audit(
        self,
        request: CitadelActionRequest,
        result: CitadelValidationResult,
    ) -> None:
        self._audit_repository.record(
            action=request.action.value,
            actor=request.actor,
            branch_name=request.branch_name,
            decision=result.decision,
            reasons=result.reasons,
            parameters=dict(request.parameters),
            override=request.override,
        )
        logger.info(
            "citadel.validation",
            extra={
                "action": request.action.value,
                "actor": request.actor,
                "branch_name": request.branch_name,
                "decision": result.decision.value,
                "allowed": result.allowed,
            },
        )


def _string_parameter(value: DecisionParameter) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None
