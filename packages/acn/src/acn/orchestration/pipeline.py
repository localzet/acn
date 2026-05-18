import logging
from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext

from acn.continual.stage import DatasetStage
from acn.controller import AdaptiveController, MetricPoint, TrainingContext
from acn.infrastructure.uow import UnitOfWork
from acn.orchestration.decision import DecisionExecutionResult, DecisionExecutor
from acn.orchestration.domain import ExperimentRecord, ExperimentStatus, StageTrainingResult
from acn.orchestration.repository import ExperimentStateRepository
from acn.orchestration.session import TrainingSession
from acn.orchestration.stage_transition import StageTransitionManager
from acn.versioning.domain import CommitRecord, Metadata
from acn.versioning.repository import TrainingVersionRepository

logger = logging.getLogger(__name__)


class EvolutionPipeline:
    def __init__(
        self,
        *,
        state_repository: ExperimentStateRepository,
        version_repository: TrainingVersionRepository,
        training_session: TrainingSession,
        controller: AdaptiveController,
        decision_executor: DecisionExecutor,
        transition_manager: StageTransitionManager,
        unit_of_work: UnitOfWork | None = None,
    ) -> None:
        self._state_repository = state_repository
        self._version_repository = version_repository
        self._training_session = training_session
        self._controller = controller
        self._decision_executor = decision_executor
        self._transition_manager = transition_manager
        self._unit_of_work = unit_of_work

    def run(
        self,
        *,
        experiment: ExperimentRecord,
        stages: Sequence[DatasetStage],
        actor: str = "orchestrator",
    ) -> ExperimentRecord:
        with self._transaction():
            self._state_repository.update_experiment(
                experiment.id,
                status=ExperimentStatus.RUNNING,
            )
        current_experiment = self._state_repository.get_experiment(experiment.id)
        metric_history: list[MetricPoint] = []
        active_execution_id: str | None = None

        try:
            for stage in stages:
                execution = self._transition_manager.start_stage(
                    experiment=current_experiment,
                    stage=stage,
                )
                active_execution_id = execution.id
                result = self._training_session.run_stage(stage)
                with self._transaction():
                    commit = self._commit_stage(
                        experiment=current_experiment,
                        stage=stage,
                        result=result,
                        actor=actor,
                    )
                    next_metric_history = [*metric_history, *result.metrics]
                    self._transition_manager.complete_stage(
                        execution_id=execution.id,
                        commit_id=commit.id,
                        metrics=_latest_metrics(result),
                    )
                    current_experiment = self._state_repository.update_experiment(
                        experiment.id,
                        current_stage_id=stage.id,
                        current_commit_id=commit.id,
                        best_commit_id=_best_commit_id(current_experiment.best_commit_id, commit),
                    )
                    decision = self._controller.decide(
                        metrics=next_metric_history,
                        context=TrainingContext(
                            branch_name=current_experiment.branch_name,
                            current_commit_id=current_experiment.current_commit_id,
                            best_commit_id=current_experiment.best_commit_id,
                        ),
                    )
                    decision_result = self._decision_executor.execute(
                        decision=decision,
                        actor=actor,
                        branch_name=current_experiment.branch_name,
                        current_commit_id=current_experiment.current_commit_id,
                    )
                    metric_history = next_metric_history
                active_execution_id = None
                self._log_decision(decision_result)

            with self._transaction():
                return self._state_repository.update_experiment(
                    experiment.id,
                    status=ExperimentStatus.COMPLETED,
                )
        except Exception:
            with self._transaction():
                if active_execution_id is not None:
                    self._transition_manager.fail_stage(active_execution_id)
                self._state_repository.update_experiment(
                    experiment.id,
                    status=ExperimentStatus.FAILED,
                )
            raise

    def _commit_stage(
        self,
        *,
        experiment: ExperimentRecord,
        stage: DatasetStage,
        result: StageTrainingResult,
        actor: str,
    ) -> CommitRecord:
        checkpoint = self._version_repository.create_checkpoint(
            uri=result.checkpoint_uri,
            content_hash=result.checkpoint_hash,
            size_bytes=result.size_bytes,
            metadata={"experiment_id": experiment.id, "stage_id": stage.id, **result.metadata},
        )
        return self._version_repository.create_commit(
            branch_name=experiment.branch_name,
            checkpoint_id=checkpoint.id,
            message=f"stage:{stage.id}",
            authored_by=actor,
            metrics=_latest_metrics(result),
            metadata={"experiment_id": experiment.id, "stage_id": stage.id},
        )

    def _log_decision(self, result: DecisionExecutionResult) -> None:
        logger.info(
            "evolution.decision_executed",
            extra={
                "action": result.action.value,
                "executed": result.executed,
                "message": result.message,
            },
        )

    def _transaction(self) -> AbstractContextManager[object]:
        if self._unit_of_work is None:
            return nullcontext()
        return self._unit_of_work.transaction()


def _latest_metrics(result: StageTrainingResult) -> Metadata:
    if not result.metrics:
        return {}
    latest = result.metrics[-1]
    return {
        "epoch": latest.epoch,
        "train_loss": latest.train_loss,
        "validation_loss": latest.validation_loss,
        "train_accuracy": latest.train_accuracy,
        "validation_accuracy": latest.validation_accuracy,
        "learning_rate": latest.learning_rate,
    }


def _best_commit_id(current_best_commit_id: str | None, commit: CommitRecord) -> str:
    return current_best_commit_id or commit.id
