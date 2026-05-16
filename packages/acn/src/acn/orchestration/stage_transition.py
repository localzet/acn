from acn.continual.stage import DatasetStage
from acn.orchestration.domain import ExperimentRecord, StageExecutionRecord, StageExecutionStatus
from acn.orchestration.repository import ExperimentStateRepository
from acn.versioning.domain import Metadata


class StageTransitionManager:
    def __init__(self, state_repository: ExperimentStateRepository) -> None:
        self._state_repository = state_repository

    def start_stage(
        self,
        *,
        experiment: ExperimentRecord,
        stage: DatasetStage,
    ) -> StageExecutionRecord:
        self._state_repository.update_experiment(
            experiment.id,
            current_stage_id=stage.id,
        )
        return self._state_repository.create_stage_execution(
            experiment_id=experiment.id,
            stage_id=stage.id,
            status=StageExecutionStatus.RUNNING,
        )

    def complete_stage(
        self,
        *,
        execution_id: str,
        commit_id: str,
        metrics: Metadata,
    ) -> StageExecutionRecord:
        return self._state_repository.update_stage_execution(
            execution_id,
            status=StageExecutionStatus.COMPLETED,
            commit_id=commit_id,
            metrics=metrics,
        )

    def fail_stage(self, execution_id: str) -> StageExecutionRecord:
        return self._state_repository.update_stage_execution(
            execution_id,
            status=StageExecutionStatus.FAILED,
        )
