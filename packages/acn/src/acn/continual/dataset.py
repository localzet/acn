from collections.abc import Callable, Sequence, Sized
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset

type ImageSample = tuple[Tensor, int]


class ClassFilteredDataset(Dataset[ImageSample]):
    def __init__(
        self,
        dataset: Dataset[Any],
        class_ids: Sequence[int],
        *,
        target_getter: Callable[[Dataset[Any]], Sequence[int]] | None = None,
    ) -> None:
        self._dataset = dataset
        self._class_ids = frozenset(class_ids)
        targets = _resolve_targets(dataset, target_getter)
        self._indices = tuple(
            index for index, target in enumerate(targets) if target in self._class_ids
        )

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> ImageSample:
        sample = self._dataset[self._indices[index]]
        image, target = parse_image_sample(sample)
        return image, target


class TransformedImageDataset(Dataset[ImageSample]):
    def __init__(
        self,
        dataset: Dataset[Any],
        transform: Callable[[Tensor], Tensor],
    ) -> None:
        self._dataset = dataset
        self._transform = transform

    def __len__(self) -> int:
        return _dataset_len(self._dataset)

    def __getitem__(self, index: int) -> ImageSample:
        image, target = parse_image_sample(self._dataset[index])
        return self._transform(image), target


class CombinedImageDataset(Dataset[ImageSample]):
    def __init__(self, datasets: Sequence[Dataset[Any]]) -> None:
        self._datasets = tuple(datasets)
        self._cumulative_sizes: list[int] = []
        total = 0
        for dataset in self._datasets:
            total += _dataset_len(dataset)
            self._cumulative_sizes.append(total)

    def __len__(self) -> int:
        return self._cumulative_sizes[-1] if self._cumulative_sizes else 0

    def __getitem__(self, index: int) -> ImageSample:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        dataset_index = 0
        while index >= self._cumulative_sizes[dataset_index]:
            dataset_index += 1
        previous_size = 0 if dataset_index == 0 else self._cumulative_sizes[dataset_index - 1]
        return parse_image_sample(self._datasets[dataset_index][index - previous_size])


def parse_image_sample(sample: object) -> ImageSample:
    if isinstance(sample, dict):
        image = sample.get("inputs")
        target = sample.get("targets")
    elif isinstance(sample, tuple | list) and len(sample) >= 2:
        image = sample[0]
        target = sample[1]
    else:
        msg = "Image sample must be a mapping or a sequence with image and target."
        raise TypeError(msg)

    if not isinstance(image, Tensor):
        msg = f"Expected image tensor, got {type(image).__name__}."
        raise TypeError(msg)
    return image, _target_to_int(target)


def _target_to_int(target: object) -> int:
    if isinstance(target, Tensor):
        return int(target.item())
    if isinstance(target, int):
        return target
    msg = f"Expected integer target, got {type(target).__name__}."
    raise TypeError(msg)


def _resolve_targets(
    dataset: Dataset[Any],
    target_getter: Callable[[Dataset[Any]], Sequence[int]] | None,
) -> Sequence[int]:
    if target_getter is not None:
        return target_getter(dataset)
    targets = getattr(dataset, "targets", None)
    if isinstance(targets, Tensor):
        return [int(value) for value in targets.tolist()]
    if isinstance(targets, list | tuple):
        return [int(value) for value in targets]

    return [parse_image_sample(dataset[index])[1] for index in range(_dataset_len(dataset))]


def _dataset_len(dataset: Dataset[Any]) -> int:
    if not isinstance(dataset, Sized):
        msg = "Dataset must implement __len__."
        raise TypeError(msg)
    return len(dataset)


def gaussian_noise_shift(severity: float) -> Callable[[Tensor], Tensor]:
    def transform(image: Tensor) -> Tensor:
        noise = torch.randn_like(image) * severity
        return torch.clamp(image + noise, 0.0, 1.0)

    return transform


def brightness_shift(delta: float) -> Callable[[Tensor], Tensor]:
    def transform(image: Tensor) -> Tensor:
        return torch.clamp(image + delta, 0.0, 1.0)

    return transform
