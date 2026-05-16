import asyncio

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from acn.citadel import CitadelSafetyLayer
from acn.continual import DatasetStage
from acn.continual.stage import DatasetSplit
from acn.controller import AdaptiveController
from acn.orchestration import (
    DecisionExecutor,
    EvolutionPipeline,
    ExperimentManager,
    RollbackCoordinator,
    SqlAlchemyExperimentStateRepository,
    StageTrainingResult,
    StageTransitionManager,
    TrainingSession,
)
from acn.orchestration.session import StageTrainingRunner
from acn.versioning.models import Base
from acn.versioning.repository import SqlAlchemyTrainingVersionRepository


class DemoRunner(StageTrainingRunner):
    async def run_stage(self, stage: DatasetStage) -> StageTrainingResult:
        return StageTrainingResult(
            checkpoint_uri=f"s3://mlflow/demo/{stage.id}.pt",
            checkpoint_hash=f"sha256:{stage.id}",
            metrics=(),
            metadata={"demo": True},
        )


async def main() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)

    with session_factory() as session:
        version_repository = SqlAlchemyTrainingVersionRepository(session)
        state_repository = SqlAlchemyExperimentStateRepository(session)
        manager = ExperimentManager(
            state_repository=state_repository,
            version_repository=version_repository,
        )
        experiment = manager.create_experiment(name="demo-lifecycle", branch_name="main")
        citadel = CitadelSafetyLayer(version_repository=version_repository)
        rollback = RollbackCoordinator(version_repository=version_repository, citadel=citadel)
        pipeline = EvolutionPipeline(
            state_repository=state_repository,
            version_repository=version_repository,
            training_session=TrainingSession(DemoRunner()),
            controller=AdaptiveController(),
            decision_executor=DecisionExecutor(
                version_repository=version_repository,
                citadel=citadel,
                rollback_coordinator=rollback,
            ),
            transition_manager=StageTransitionManager(state_repository),
        )
        stages = (
            DatasetStage(
                id="stage-0",
                source_name="fashion-mnist",
                class_ids=(0, 1),
                split=DatasetSplit.TRAIN,
                introduced_class_ids=(0, 1),
            ),
            DatasetStage(
                id="stage-1",
                source_name="fashion-mnist",
                class_ids=(2, 3),
                split=DatasetSplit.TRAIN,
                introduced_class_ids=(2, 3),
                replay_ratio=0.25,
            ),
        )

        completed = await pipeline.run(experiment=experiment, stages=stages)
        history = version_repository.list_branch_history("main")
        executions = state_repository.list_stage_executions(experiment.id)

        print(f"experiment={completed.name} status={completed.status.value}")
        print(f"commits={len(history)} stages={len(executions)}")


if __name__ == "__main__":
    asyncio.run(main())
