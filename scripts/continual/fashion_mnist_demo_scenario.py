from pathlib import Path

from acn.continual import (
    ContinualLearningScenario,
    DatasetStageConfig,
    ReplayBuffer,
    ReplayBufferConfig,
    SyntheticDomainShiftSource,
)
from acn.continual.torchvision_sources import build_fashion_mnist_source


def main() -> None:
    source = build_fashion_mnist_source(Path("data"))
    shifted_source = SyntheticDomainShiftSource(
        name="fashion-mnist-noisy",
        base_source=source,
        shift_name="gaussian_noise",
        severity=0.12,
    )
    scenario = ContinualLearningScenario.from_configs(
        scenario_id="fashion-mnist-demo",
        sources={
            source.name: source,
            shifted_source.name: shifted_source,
        },
        stage_configs=[
            DatasetStageConfig(id="stage-0", source_name=source.name, class_ids=(0, 1)),
            DatasetStageConfig(
                id="stage-1",
                source_name=source.name,
                class_ids=(2, 3),
                replay_ratio=0.25,
            ),
            DatasetStageConfig(
                id="stage-2-shift",
                source_name=source.name,
                domain_shift=shifted_source.name,
                class_ids=(0, 1, 2, 3),
                replay_ratio=0.2,
            ),
        ],
    )
    replay_buffer = ReplayBuffer(ReplayBufferConfig(capacity=256, samples_per_class=32))

    for stage in scenario.stages:
        dataset = scenario.build_stage_dataset_with_replay(stage, replay_buffer=replay_buffer)
        print(
            f"stage={stage.id} classes={stage.class_ids} "
            f"introduced={stage.introduced_class_ids} samples={len(dataset)}"
        )
        replay_buffer.add_dataset(scenario.build_stage_dataset(stage))


if __name__ == "__main__":
    main()
