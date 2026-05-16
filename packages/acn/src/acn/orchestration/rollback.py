from acn.citadel import CitadelActionRequest, CitadelSafetyLayer
from acn.controller import AdaptiveAction
from acn.versioning.domain import BranchRecord
from acn.versioning.repository import TrainingVersionRepository


class RollbackCoordinator:
    def __init__(
        self,
        *,
        version_repository: TrainingVersionRepository,
        citadel: CitadelSafetyLayer,
    ) -> None:
        self._version_repository = version_repository
        self._citadel = citadel

    def rollback(
        self,
        *,
        actor: str,
        branch_name: str,
        current_commit_id: str | None,
        target_commit_id: str,
    ) -> BranchRecord:
        result = self._citadel.validate_action(
            CitadelActionRequest(
                action=AdaptiveAction.ROLLBACK,
                actor=actor,
                branch_name=branch_name,
                current_commit_id=current_commit_id,
                parameters={"target_commit_id": target_commit_id},
            )
        )
        if not result.allowed:
            msg = "; ".join(result.reasons)
            raise RuntimeError(msg)

        return self._version_repository.rollback_branch(
            branch_name=branch_name,
            target_commit_id=target_commit_id,
        )
