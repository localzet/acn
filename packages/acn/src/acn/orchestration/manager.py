from acn.orchestration.domain import ExperimentRecord, ExperimentStatus
from acn.orchestration.repository import ExperimentStateRepository
from acn.versioning.domain import Metadata
from acn.versioning.exceptions import BranchNotFoundError
from acn.versioning.repository import TrainingVersionRepository


class ExperimentManager:
    def __init__(
        self,
        *,
        state_repository: ExperimentStateRepository,
        version_repository: TrainingVersionRepository,
    ) -> None:
        self._state_repository = state_repository
        self._version_repository = version_repository

    def create_experiment(
        self,
        *,
        name: str,
        branch_name: str = "main",
        metadata: Metadata | None = None,
        experiment_id: str | None = None,
    ) -> ExperimentRecord:
        try:
            self._version_repository.get_branch(branch_name)
        except BranchNotFoundError:
            self._version_repository.create_branch(name=branch_name)

        return self._state_repository.create_experiment(
            name=name,
            branch_name=branch_name,
            metadata=metadata,
            experiment_id=experiment_id,
        )

    def start(self, experiment_id: str) -> ExperimentRecord:
        return self._state_repository.update_experiment(
            experiment_id,
            status=ExperimentStatus.RUNNING,
        )

    def complete(self, experiment_id: str) -> ExperimentRecord:
        return self._state_repository.update_experiment(
            experiment_id,
            status=ExperimentStatus.COMPLETED,
        )

    def fail(self, experiment_id: str) -> ExperimentRecord:
        return self._state_repository.update_experiment(
            experiment_id,
            status=ExperimentStatus.FAILED,
        )
