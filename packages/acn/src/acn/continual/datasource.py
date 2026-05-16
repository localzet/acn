from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from torch import Tensor
from torch.utils.data import Dataset

from acn.continual.dataset import (
    ClassFilteredDataset,
    TransformedImageDataset,
    brightness_shift,
    gaussian_noise_shift,
)
from acn.continual.stage import DatasetSplit


class IDataSource(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def class_ids(self) -> tuple[int, ...]: ...

    def build_dataset(
        self,
        *,
        split: DatasetSplit,
        class_ids: Sequence[int] | None = None,
    ) -> Dataset[Any]: ...


@dataclass(frozen=True, slots=True)
class ImageDatasetSource:
    name: str
    class_ids: tuple[int, ...]
    dataset_factory: Callable[[DatasetSplit], Dataset[Any]]
    target_getter: Callable[[Dataset[Any]], Sequence[int]] | None = None
    root: Path | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def build_dataset(
        self,
        *,
        split: DatasetSplit,
        class_ids: Sequence[int] | None = None,
    ) -> Dataset[Any]:
        dataset = self.dataset_factory(split)
        if class_ids is None:
            return dataset
        return ClassFilteredDataset(dataset, class_ids, target_getter=self.target_getter)


@dataclass(frozen=True, slots=True)
class SyntheticDomainShiftSource:
    name: str
    base_source: IDataSource
    shift_name: str = "gaussian_noise"
    severity: float = 0.15

    @property
    def class_ids(self) -> tuple[int, ...]:
        return self.base_source.class_ids

    def build_dataset(
        self,
        *,
        split: DatasetSplit,
        class_ids: Sequence[int] | None = None,
    ) -> Dataset[Any]:
        dataset = self.base_source.build_dataset(split=split, class_ids=class_ids)
        return TransformedImageDataset(dataset, _build_shift(self.shift_name, self.severity))


def _build_shift(shift_name: str, severity: float) -> Callable[[Tensor], Tensor]:
    if shift_name == "gaussian_noise":
        return gaussian_noise_shift(severity)
    if shift_name == "brightness":
        return brightness_shift(severity)

    msg = f"Unsupported synthetic domain shift: {shift_name}"
    raise ValueError(msg)
