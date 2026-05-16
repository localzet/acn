from collections import defaultdict
from collections.abc import Sized
from dataclasses import dataclass
from random import Random
from typing import Any

from torch.utils.data import Dataset

from acn.continual.dataset import ImageSample, parse_image_sample


@dataclass(frozen=True, slots=True)
class ReplayBufferConfig:
    capacity: int
    samples_per_class: int | None = None
    seed: int = 13


class ReplayBuffer:
    def __init__(self, config: ReplayBufferConfig) -> None:
        if config.capacity <= 0:
            msg = "Replay buffer capacity must be positive."
            raise ValueError(msg)
        self._config = config
        self._samples: list[ImageSample] = []
        self._rng = Random(config.seed)  # noqa: S311

    def __len__(self) -> int:
        return len(self._samples)

    def add_dataset(self, dataset: Dataset[Any]) -> None:
        candidates = [parse_image_sample(dataset[index]) for index in range(_dataset_len(dataset))]
        if self._config.samples_per_class is not None:
            candidates = _balanced_sample(
                candidates,
                samples_per_class=self._config.samples_per_class,
                rng=self._rng,
            )

        for image, target in candidates:
            self._samples.append((image.detach().cpu().clone(), target))

        if len(self._samples) > self._config.capacity:
            self._samples = self._rng.sample(self._samples, self._config.capacity)

    def as_dataset(self, *, max_samples: int | None = None) -> Dataset[ImageSample]:
        samples = self._samples
        if max_samples is not None and max_samples < len(samples):
            samples = self._rng.sample(samples, max_samples)
        return ReplayDataset(samples)

    def class_counts(self) -> dict[int, int]:
        counts: dict[int, int] = defaultdict(int)
        for _image, target in self._samples:
            counts[target] += 1
        return dict(counts)


class ReplayDataset(Dataset[ImageSample]):
    def __init__(self, samples: list[ImageSample]) -> None:
        self._samples = tuple(samples)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> ImageSample:
        image, target = self._samples[index]
        return image.clone(), target


def _balanced_sample(
    samples: list[ImageSample],
    *,
    samples_per_class: int,
    rng: Random,
) -> list[ImageSample]:
    by_class: dict[int, list[ImageSample]] = defaultdict(list)
    for sample in samples:
        by_class[sample[1]].append(sample)

    selected: list[ImageSample] = []
    for class_samples in by_class.values():
        if len(class_samples) <= samples_per_class:
            selected.extend(class_samples)
        else:
            selected.extend(rng.sample(class_samples, samples_per_class))
    return selected


def _dataset_len(dataset: Dataset[Any]) -> int:
    if not isinstance(dataset, Sized):
        msg = "Dataset must implement __len__."
        raise TypeError(msg)
    return len(dataset)
