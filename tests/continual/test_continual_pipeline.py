from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from acn.continual import (
    ContinualEvaluationPipeline,
    ContinualLearningScenario,
    DatasetStageConfig,
    ForgettingEvaluator,
    ImageDatasetSource,
    ReplayBuffer,
    ReplayBufferConfig,
    SyntheticDomainShiftSource,
)
from acn.continual.stage import DatasetSplit


class ToyImageDataset(Dataset[tuple[Tensor, int]]):
    def __init__(self, targets: Sequence[int]) -> None:
        self.targets = list(targets)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        target = self.targets[index]
        image = torch.full((1, 2, 2), fill_value=float(target) / 10.0)
        return image, target


class ConstantClassifier(nn.Module):
    def __init__(self, predictions: Sequence[int], class_count: int = 4) -> None:
        super().__init__()
        self._predictions = list(predictions)
        self._class_count = class_count
        self._offset = 0

    def forward(self, inputs: Tensor) -> Tensor:
        batch_size = int(inputs.size(0))
        logits = torch.zeros(batch_size, self._class_count)
        batch_predictions = self._predictions[self._offset : self._offset + batch_size]
        for index, prediction in enumerate(batch_predictions):
            logits[index, prediction] = 1.0
        self._offset += batch_size
        return logits


def _source(name: str = "toy") -> ImageDatasetSource:
    return ImageDatasetSource(
        name=name,
        class_ids=(0, 1, 2, 3),
        dataset_factory=lambda _split: ToyImageDataset([0, 0, 1, 1, 2, 2, 3, 3]),
    )


def test_image_dataset_source_filters_classes() -> None:
    source = _source()

    dataset = source.build_dataset(split=DatasetSplit.TRAIN, class_ids=(1, 3))

    assert len(dataset) == 4
    assert [dataset[index][1] for index in range(len(dataset))] == [1, 1, 3, 3]


def test_synthetic_domain_shift_changes_images() -> None:
    source = _source()
    shifted_source = SyntheticDomainShiftSource(
        name="toy-bright",
        base_source=source,
        shift_name="brightness",
        severity=0.2,
    )

    original_dataset = source.build_dataset(split=DatasetSplit.TRAIN, class_ids=(1,))
    shifted_dataset = shifted_source.build_dataset(split=DatasetSplit.TRAIN, class_ids=(1,))

    assert torch.allclose(
        shifted_dataset[0][0],
        torch.clamp(original_dataset[0][0] + 0.2, 0.0, 1.0),
    )
    assert shifted_dataset[0][1] == original_dataset[0][1]


def test_scenario_tracks_incremental_class_introduction() -> None:
    source = _source()
    scenario = ContinualLearningScenario.from_configs(
        scenario_id="toy-incremental",
        sources={"toy": source},
        stage_configs=[
            DatasetStageConfig(id="s1", source_name="toy", class_ids=(0, 1)),
            DatasetStageConfig(id="s2", source_name="toy", class_ids=(1, 2)),
            DatasetStageConfig(id="s3", source_name="toy", class_ids=(2, 3)),
        ],
    )

    assert scenario.stages[0].introduced_class_ids == (0, 1)
    assert scenario.stages[1].introduced_class_ids == (2,)
    assert scenario.stages[2].introduced_class_ids == (3,)
    assert scenario.old_class_ids_before(scenario.stages[2]) == (0, 1, 2)


def test_scenario_combines_stage_dataset_with_replay() -> None:
    source = _source()
    scenario = ContinualLearningScenario.from_configs(
        scenario_id="toy-replay",
        sources={"toy": source},
        stage_configs=[
            DatasetStageConfig(id="s1", source_name="toy", class_ids=(0, 1)),
            DatasetStageConfig(id="s2", source_name="toy", class_ids=(2, 3), replay_ratio=0.5),
        ],
    )
    replay = ReplayBuffer(ReplayBufferConfig(capacity=10, samples_per_class=1))
    replay.add_dataset(scenario.build_stage_dataset(scenario.stages[0]))

    combined = scenario.build_stage_dataset_with_replay(scenario.stages[1], replay_buffer=replay)

    assert len(replay) == 2
    assert len(combined) == 6
    assert replay.class_counts() == {0: 1, 1: 1}


def test_forgetting_evaluator_calculates_continual_metrics() -> None:
    evaluator = ForgettingEvaluator(adaptation_threshold=0.75)
    first = evaluator.evaluate_predictions(
        stage_id="s1",
        introduced_class_ids=(0, 1),
        old_class_ids=(),
        targets=[0, 0, 1, 1],
        predictions=[0, 0, 1, 0],
    )
    second = evaluator.evaluate_predictions(
        stage_id="s2",
        introduced_class_ids=(2,),
        old_class_ids=(0, 1),
        targets=[0, 0, 1, 1, 2, 2],
        predictions=[0, 1, 1, 0, 2, 2],
    )

    assert first.new_class_adaptation == 0.75
    assert second.old_class_retention == 0.5
    assert second.new_class_adaptation == 1.0
    assert second.forgetting_score == 0.25
    assert second.adaptation_latency == 0


def test_evaluation_pipeline_evaluates_model_predictions() -> None:
    dataset = ToyImageDataset([0, 0, 1, 1])
    dataloader = DataLoader(dataset, batch_size=2)
    model = ConstantClassifier([0, 1, 1, 1], class_count=2)
    pipeline = ContinualEvaluationPipeline(ForgettingEvaluator(adaptation_threshold=0.5))

    metrics = pipeline.evaluate_model(
        model=model,
        dataloader=dataloader,
        stage_id="s1",
        introduced_class_ids=(0, 1),
        old_class_ids=(),
    )

    assert metrics.per_class_accuracy == {0: 0.5, 1: 1.0}
    assert metrics.new_class_adaptation == 0.75
