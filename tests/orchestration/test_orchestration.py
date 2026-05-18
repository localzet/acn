from sqlalchemy.orm import Session

from acn.citadel import CitadelSafetyLayer, SqlAlchemyAuditLogRepository
from acn.continual import DatasetStage, DatasetStageConfig
from acn.continual.stage import DatasetSplit
from acn.controller import AdaptiveAction, AdaptiveController, ControllerDecision, ControllerSignals
from acn.orchestration import (
    DecisionExecutor,
    EvolutionPipeline,
    ExperimentManager,
    ExperimentStatus,
    InMemoryExperimentStateRepository,
    RollbackCoordinator,
    SqlAlchemyExperimentStateRepository,
    StageExecutionStatus,
    StageTrainingResult,
    StageTransitionManager,
    TrainingSession,
)
from acn.orchestration.session import StageTrainingRunner
from acn.versioning.repository import SqlAlchemyTrainingVersionRepository


class StaticRunner(StageTrainingRunner):
    def run_stage(self, stage: DatasetStage) -> StageTrainingResult:
        return StageTrainingResult(
            checkpoint_uri=f"s3://mlflow/{stage.id}.pt",
            checkpoint_hash=f"sha:{stage.id}",
            metrics=(),
            metadata={"stage_id": stage.id},
        )


class StaticController(AdaptiveController):
    def decide(self, **_kwargs: object) -> ControllerDecision:
        return ControllerDecision(
            action=AdaptiveAction.CONTINUE_TRAINING,
            confidence=1.0,
            reasons=("test",),
            signals=ControllerSignals(),
        )


def _stage(stage_id: str) -> DatasetStage:
    return DatasetStage(
        id=stage_id,
        source_name="toy",
        class_ids=(0, 1),
        split=DatasetSplit.TRAIN,
        introduced_class_ids=(0, 1),
    )


def _repositories(
    session: Session,
) -> tuple[SqlAlchemyTrainingVersionRepository, SqlAlchemyExperimentStateRepository]:
    return (
        SqlAlchemyTrainingVersionRepository(session),
        SqlAlchemyExperimentStateRepository(session),
    )


def test_experiment_manager_creates_branch_and_persists_experiment(session: Session) -> None:
    version_repository, state_repository = _repositories(session)
    manager = ExperimentManager(
        state_repository=state_repository,
        version_repository=version_repository,
    )

    experiment = manager.create_experiment(
        name="continual-demo",
        branch_name="main",
        experiment_id="exp_1",
    )
    started = manager.start(experiment.id)

    assert experiment.status is ExperimentStatus.CREATED
    assert started.status is ExperimentStatus.RUNNING
    assert version_repository.get_branch("main").name == "main"


def test_stage_transition_manager_records_stage_lifecycle() -> None:
    state_repository = InMemoryExperimentStateRepository()
    experiment = state_repository.create_experiment(name="demo", branch_name="main")
    manager = StageTransitionManager(state_repository)

    execution = manager.start_stage(experiment=experiment, stage=_stage("s1"))
    completed = manager.complete_stage(
        execution_id=execution.id,
        commit_id="cmt_1",
        metrics={"validation_loss": 0.5},
    )

    assert execution.status is StageExecutionStatus.RUNNING
    assert completed.status is StageExecutionStatus.COMPLETED
    assert completed.commit_id == "cmt_1"


def test_decision_executor_creates_experimental_branch(session: Session) -> None:
    version_repository, _state_repository = _repositories(session)
    version_repository.create_branch(name="main")
    checkpoint = version_repository.create_checkpoint(
        uri="s3://mlflow/base.pt",
        content_hash="sha:base",
    )
    commit = version_repository.create_commit(
        branch_name="main",
        checkpoint_id=checkpoint.id,
        message="base",
        commit_id="cmt_base",
    )
    citadel = CitadelSafetyLayer(version_repository=version_repository)
    rollback = RollbackCoordinator(version_repository=version_repository, citadel=citadel)
    executor = DecisionExecutor(
        version_repository=version_repository,
        citadel=citadel,
        rollback_coordinator=rollback,
    )

    result = executor.execute(
        decision=ControllerDecision(
            action=AdaptiveAction.CREATE_EXPERIMENTAL_BRANCH,
            confidence=0.8,
            reasons=("plateau",),
            signals=ControllerSignals(plateau=True),
            parameters={"source_commit_id": commit.id},
        ),
        actor="controller",
        branch_name="main",
        current_commit_id=commit.id,
    )

    assert result.executed
    assert result.metadata["base_commit_id"] == "cmt_base"


def test_rollback_coordinator_routes_through_citadel(session: Session) -> None:
    version_repository, _state_repository = _repositories(session)
    version_repository.create_branch(name="main")
    first_checkpoint = version_repository.create_checkpoint(
        uri="s3://mlflow/a.pt",
        content_hash="sha:a",
    )
    second_checkpoint = version_repository.create_checkpoint(
        uri="s3://mlflow/b.pt",
        content_hash="sha:b",
    )
    first = version_repository.create_commit(
        branch_name="main",
        checkpoint_id=first_checkpoint.id,
        message="first",
        commit_id="cmt_first",
    )
    second = version_repository.create_commit(
        branch_name="main",
        checkpoint_id=second_checkpoint.id,
        message="second",
        commit_id="cmt_second",
    )
    citadel = CitadelSafetyLayer(
        version_repository=version_repository,
        audit_repository=SqlAlchemyAuditLogRepository(session),
    )
    coordinator = RollbackCoordinator(version_repository=version_repository, citadel=citadel)

    branch = coordinator.rollback(
        actor="controller",
        branch_name="main",
        current_commit_id=second.id,
        target_commit_id=first.id,
    )

    assert branch.head_commit_id == first.id


def test_evolution_pipeline_runs_stages_and_commits(session: Session) -> None:
    version_repository, state_repository = _repositories(session)
    manager = ExperimentManager(
        state_repository=state_repository,
        version_repository=version_repository,
    )
    experiment = manager.create_experiment(name="pipeline", branch_name="main")
    citadel = CitadelSafetyLayer(version_repository=version_repository)
    rollback = RollbackCoordinator(version_repository=version_repository, citadel=citadel)
    pipeline = EvolutionPipeline(
        state_repository=state_repository,
        version_repository=version_repository,
        training_session=TrainingSession(StaticRunner()),
        controller=StaticController(),
        decision_executor=DecisionExecutor(
            version_repository=version_repository,
            citadel=citadel,
            rollback_coordinator=rollback,
        ),
        transition_manager=StageTransitionManager(state_repository),
    )

    completed = pipeline.run(
        experiment=experiment,
        stages=(_stage("s1"), _stage("s2")),
    )

    history = version_repository.list_branch_history("main")
    executions = state_repository.list_stage_executions(experiment.id)

    assert completed.status is ExperimentStatus.COMPLETED
    assert len(history) == 2
    assert len(executions) == 2
    assert all(execution.status is StageExecutionStatus.COMPLETED for execution in executions)


def test_stage_config_remains_usable_for_orchestration() -> None:
    config = DatasetStageConfig(id="s1", source_name="fashion-mnist", class_ids=(0, 1))

    assert config.id == "s1"
