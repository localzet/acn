from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine, event, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from acn.citadel import CitadelSafetyLayer
from acn.continual.stage import DatasetSplit, DatasetStage
from acn.controller import AdaptiveAction, AdaptiveController, ControllerDecision, ControllerSignals
from acn.infrastructure import SqlAlchemySessionUnitOfWork
from acn.orchestration import (
    DecisionExecutor,
    EvolutionPipeline,
    ExperimentManager,
    ExperimentStatus,
    RollbackCoordinator,
    SqlAlchemyExperimentStateRepository,
    StageTrainingResult,
    StageTransitionManager,
    TrainingSession,
)
from acn.orchestration.models import StageExecutionModel
from acn.orchestration.session import StageTrainingRunner
from acn.versioning.exceptions import BranchHeadConflictError
from acn.versioning.models import Base, BranchModel, CommitModel, StableCheckpointModel
from acn.versioning.repository import SqlAlchemyTrainingVersionRepository


@pytest.fixture
def fk_session() -> Iterator[Session]:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection: object, _connection_record: object) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    with factory() as session:
        yield session


class StaticRunner(StageTrainingRunner):
    def run_stage(self, stage: DatasetStage) -> StageTrainingResult:
        return StageTrainingResult(
            checkpoint_uri=f"file:///tmp/{stage.id}.pt",
            checkpoint_hash=f"sha256:{stage.id}",
            metrics=(),
            metadata={"stage_id": stage.id},
        )


class DuplicateCheckpointRunner(StageTrainingRunner):
    def run_stage(self, stage: DatasetStage) -> StageTrainingResult:
        return StageTrainingResult(
            checkpoint_uri=f"file:///tmp/{stage.id}.pt",
            checkpoint_hash="sha256:duplicate",
            metrics=(),
            metadata={"stage_id": stage.id},
        )


class InvalidCommitController(AdaptiveController):
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
        class_ids=(0,),
        split=DatasetSplit.TRAIN,
        introduced_class_ids=(0,),
    )


def _count(session: Session, model: type[object]) -> int:
    return int(session.scalar(select(func.count()).select_from(model)) or 0)


def test_experiment_commit_foreign_keys_are_enforced(fk_session: Session) -> None:
    state_repository = SqlAlchemyExperimentStateRepository(fk_session)
    experiment = state_repository.create_experiment(name="fk", branch_name="main")

    with pytest.raises(IntegrityError):
        state_repository.update_experiment(
            experiment.id,
            current_commit_id="missing",
        )


def test_stage_execution_commit_foreign_key_is_enforced(fk_session: Session) -> None:
    state_repository = SqlAlchemyExperimentStateRepository(fk_session)
    experiment = state_repository.create_experiment(name="stage-fk", branch_name="main")
    execution = state_repository.create_stage_execution(
        experiment_id=experiment.id,
        stage_id="s1",
    )

    with pytest.raises(IntegrityError):
        state_repository.update_stage_execution(
            execution.id,
            status=execution.status,
            commit_id="missing",
        )


def test_unit_of_work_rolls_back_orchestration_mutations(fk_session: Session) -> None:
    version_repository = SqlAlchemyTrainingVersionRepository(fk_session)
    state_repository = SqlAlchemyExperimentStateRepository(fk_session)
    version_repository.create_branch(name="main")
    experiment = state_repository.create_experiment(name="uow", branch_name="main")

    with pytest.raises(IntegrityError), SqlAlchemySessionUnitOfWork(fk_session).transaction():
        checkpoint = version_repository.create_checkpoint(
            uri="file:///tmp/uow.pt",
            content_hash="sha256:uow",
        )
        version_repository.create_commit(
            branch_name="main",
            checkpoint_id=checkpoint.id,
            message="created-inside-uow",
        )
        state_repository.update_experiment(
            experiment.id,
            current_commit_id="missing",
        )

    assert _count(fk_session, StableCheckpointModel) == 0
    assert _count(fk_session, CommitModel) == 0
    assert fk_session.scalar(select(BranchModel).where(BranchModel.name == "main")) is not None


def test_pipeline_transaction_prevents_orphaned_stage_state(fk_session: Session) -> None:
    version_repository = SqlAlchemyTrainingVersionRepository(fk_session)
    state_repository = SqlAlchemyExperimentStateRepository(fk_session)
    manager = ExperimentManager(
        state_repository=state_repository,
        version_repository=version_repository,
    )
    experiment = manager.create_experiment(name="pipeline-uow", branch_name="main")
    citadel = CitadelSafetyLayer(version_repository=version_repository)
    rollback = RollbackCoordinator(version_repository=version_repository, citadel=citadel)
    pipeline = EvolutionPipeline(
        state_repository=state_repository,
        version_repository=version_repository,
        training_session=TrainingSession(StaticRunner()),
        controller=InvalidCommitController(),
        decision_executor=DecisionExecutor(
            version_repository=version_repository,
            citadel=citadel,
            rollback_coordinator=rollback,
        ),
        transition_manager=StageTransitionManager(state_repository),
        unit_of_work=SqlAlchemySessionUnitOfWork(fk_session),
    )

    completed = pipeline.run(experiment=experiment, stages=(_stage("s1"),))

    assert completed.status is ExperimentStatus.COMPLETED
    assert _count(fk_session, CommitModel) == 1
    assert _count(fk_session, StageExecutionModel) == 1
    assert state_repository.list_stage_executions(experiment.id)[0].commit_id is not None


def test_failed_commit_creation_marks_stage_failed_without_partial_commit(
    fk_session: Session,
) -> None:
    version_repository = SqlAlchemyTrainingVersionRepository(fk_session)
    state_repository = SqlAlchemyExperimentStateRepository(fk_session)
    manager = ExperimentManager(
        state_repository=state_repository,
        version_repository=version_repository,
    )
    experiment = manager.create_experiment(name="pipeline-failure", branch_name="main")
    citadel = CitadelSafetyLayer(version_repository=version_repository)
    rollback = RollbackCoordinator(version_repository=version_repository, citadel=citadel)
    pipeline = EvolutionPipeline(
        state_repository=state_repository,
        version_repository=version_repository,
        training_session=TrainingSession(DuplicateCheckpointRunner()),
        controller=InvalidCommitController(),
        decision_executor=DecisionExecutor(
            version_repository=version_repository,
            citadel=citadel,
            rollback_coordinator=rollback,
        ),
        transition_manager=StageTransitionManager(state_repository),
        unit_of_work=SqlAlchemySessionUnitOfWork(fk_session),
    )

    with pytest.raises(ValueError, match="Checkpoint URI and content hash"):
        pipeline.run(experiment=experiment, stages=(_stage("s1"), _stage("s2")))

    executions = state_repository.list_stage_executions(experiment.id)

    assert _count(fk_session, CommitModel) == 1
    assert state_repository.get_experiment(experiment.id).status is ExperimentStatus.FAILED
    assert [execution.status.value for execution in executions] == ["completed", "failed"]


def test_stale_branch_head_update_is_rejected(fk_session: Session) -> None:
    repository = SqlAlchemyTrainingVersionRepository(fk_session)
    repository.create_branch(name="main")
    first_checkpoint = repository.create_checkpoint(uri="memory://a", content_hash="sha256:a")
    second_checkpoint = repository.create_checkpoint(uri="memory://b", content_hash="sha256:b")
    third_checkpoint = repository.create_checkpoint(uri="memory://c", content_hash="sha256:c")
    first = repository.create_commit(
        branch_name="main",
        checkpoint_id=first_checkpoint.id,
        message="first",
    )
    second = repository.create_commit(
        branch_name="main",
        checkpoint_id=second_checkpoint.id,
        message="second",
    )
    third = repository.create_commit(
        branch_name="main",
        checkpoint_id=third_checkpoint.id,
        message="third",
    )

    with pytest.raises(BranchHeadConflictError):
        repository.rollback_branch(
            branch_name="main",
            target_commit_id=first.id,
            expected_head_commit_id=second.id,
        )

    assert repository.get_branch("main").head_commit_id == third.id
