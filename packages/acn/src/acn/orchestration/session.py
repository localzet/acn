from typing import Protocol

from acn.continual.stage import DatasetStage
from acn.orchestration.domain import StageTrainingResult


class StageTrainingRunner(Protocol):
    def run_stage(self, stage: DatasetStage) -> StageTrainingResult: ...


class TrainingSession:
    def __init__(self, runner: StageTrainingRunner) -> None:
        self._runner = runner

    def run_stage(self, stage: DatasetStage) -> StageTrainingResult:
        return self._runner.run_stage(stage)
