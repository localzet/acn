from dataclasses import dataclass
from typing import Any, Protocol

from acn.artifacts import ArtifactStore
from acn.citadel import CitadelActionRequest, CitadelSafetyLayer
from acn.controller import AdaptiveAction
from acn.training.config import CheckpointState
from acn.versioning.domain import BranchRecord, CommitRecord, StableCheckpointRecord
from acn.versioning.repository import TrainingVersionRepository


class RollbackRestorationError(RuntimeError):
    """Raised when rollback restoration cannot be started safely."""


class StateLoadable(Protocol):
    def load_state_dict(self, state_dict: dict[str, Any]) -> object: ...


@dataclass(frozen=True, slots=True)
class RollbackRestorationResult:
    branch: BranchRecord
    commit: CommitRecord
    checkpoint: StableCheckpointRecord
    state: CheckpointState


class RollbackCoordinator:
    def __init__(
        self,
        *,
        version_repository: TrainingVersionRepository,
        citadel: CitadelSafetyLayer,
        artifact_store: ArtifactStore | None = None,
    ) -> None:
        self._version_repository = version_repository
        self._citadel = citadel
        self._artifact_store = artifact_store

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
            expected_head_commit_id=current_commit_id,
        )

    def rollback_and_restore(
        self,
        *,
        actor: str,
        branch_name: str,
        current_commit_id: str | None,
        target_commit_id: str,
        model: StateLoadable,
        optimizer: StateLoadable | None = None,
        scheduler: StateLoadable | None = None,
        scaler: StateLoadable | None = None,
        artifact_store: ArtifactStore | None = None,
        map_location: str = "cpu",
    ) -> RollbackRestorationResult:
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

        store = artifact_store or self._artifact_store
        if store is None:
            msg = "Rollback restoration requires an artifact store."
            raise RollbackRestorationError(msg)

        commit = self._version_repository.get_commit(target_commit_id)
        checkpoint = self._version_repository.get_checkpoint(commit.checkpoint_id)
        payload = store.load_checkpoint(
            checkpoint.uri,
            expected_checksum=checkpoint.content_hash,
            map_location=map_location,
        )

        model.load_state_dict(payload["model_state"])
        if optimizer is not None:
            optimizer.load_state_dict(payload["optimizer_state"])
        if scheduler is not None and payload["scheduler_state"] is not None:
            scheduler.load_state_dict(payload["scheduler_state"])
        if scaler is not None and payload["scaler_state"] is not None:
            scaler.load_state_dict(payload["scaler_state"])

        branch = self._version_repository.rollback_branch(
            branch_name=branch_name,
            target_commit_id=target_commit_id,
            expected_head_commit_id=current_commit_id,
        )
        return RollbackRestorationResult(
            branch=branch,
            commit=commit,
            checkpoint=checkpoint,
            state=CheckpointState(
                epoch=payload["epoch"],
                global_step=payload["global_step"],
                best_validation_loss=payload["best_validation_loss"],
            ),
        )
