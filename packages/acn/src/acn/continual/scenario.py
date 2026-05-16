from collections.abc import Mapping, Sequence, Sized
from dataclasses import dataclass
from typing import Any

from torch.utils.data import Dataset

from acn.continual.dataset import CombinedImageDataset
from acn.continual.datasource import IDataSource
from acn.continual.replay import ReplayBuffer
from acn.continual.stage import DatasetSplit, DatasetStage, DatasetStageConfig


@dataclass(frozen=True, slots=True)
class ContinualLearningScenario:
    id: str
    stages: tuple[DatasetStage, ...]
    sources: Mapping[str, IDataSource]

    @classmethod
    def from_configs(
        cls,
        *,
        scenario_id: str,
        stage_configs: Sequence[DatasetStageConfig],
        sources: Mapping[str, IDataSource],
    ) -> "ContinualLearningScenario":
        introduced: set[int] = set()
        stages: list[DatasetStage] = []
        for config in stage_configs:
            if config.source_name not in sources:
                msg = f"Unknown data source for stage {config.id}: {config.source_name}"
                raise ValueError(msg)
            introduced_class_ids = tuple(
                class_id for class_id in config.class_ids if class_id not in introduced
            )
            introduced.update(config.class_ids)
            stages.append(
                DatasetStage(
                    id=config.id,
                    source_name=config.source_name,
                    class_ids=config.class_ids,
                    split=config.split,
                    introduced_class_ids=introduced_class_ids,
                    domain_shift=config.domain_shift,
                    replay_ratio=config.replay_ratio,
                    metadata=config.metadata,
                )
            )

        return cls(id=scenario_id, stages=tuple(stages), sources=sources)

    def build_stage_dataset(
        self,
        stage: DatasetStage,
        *,
        split: DatasetSplit | None = None,
    ) -> Dataset[Any]:
        source_name = stage.domain_shift or stage.source_name
        source = self.sources[source_name]
        return source.build_dataset(split=split or stage.split, class_ids=stage.class_ids)

    def build_stage_dataset_with_replay(
        self,
        stage: DatasetStage,
        *,
        replay_buffer: ReplayBuffer,
        split: DatasetSplit | None = None,
    ) -> Dataset[Any]:
        stage_dataset = self.build_stage_dataset(stage, split=split)
        replay_count = int(_dataset_len(stage_dataset) * stage.replay_ratio)
        if replay_count <= 0 or len(replay_buffer) == 0:
            return stage_dataset
        return CombinedImageDataset(
            [stage_dataset, replay_buffer.as_dataset(max_samples=replay_count)]
        )

    def old_class_ids_before(self, stage: DatasetStage) -> tuple[int, ...]:
        old_classes: set[int] = set()
        for candidate in self.stages:
            if candidate.id == stage.id:
                break
            old_classes.update(candidate.class_ids)
        return tuple(sorted(old_classes))


def _dataset_len(dataset: Dataset[Any]) -> int:
    if not isinstance(dataset, Sized):
        msg = "Dataset must implement __len__."
        raise TypeError(msg)
    return len(dataset)
