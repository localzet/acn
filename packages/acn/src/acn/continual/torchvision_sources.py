from collections.abc import Callable
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset
from torchvision import datasets, transforms  # type: ignore[import-untyped]

from acn.continual.datasource import ImageDatasetSource
from acn.continual.stage import DatasetSplit


def build_fashion_mnist_source(data_dir: Path) -> ImageDatasetSource:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ]
    )
    return ImageDatasetSource(
        name="fashion-mnist",
        class_ids=tuple(range(10)),
        dataset_factory=_vision_factory(
            lambda split: datasets.FashionMNIST(
                root=data_dir,
                train=split is DatasetSplit.TRAIN,
                transform=transform,
                download=True,
            )
        ),
        root=data_dir,
    )


def build_cifar10_source(data_dir: Path) -> ImageDatasetSource:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    return ImageDatasetSource(
        name="cifar10",
        class_ids=tuple(range(10)),
        dataset_factory=_vision_factory(
            lambda split: datasets.CIFAR10(
                root=data_dir,
                train=split is DatasetSplit.TRAIN,
                transform=transform,
                download=True,
            )
        ),
        root=data_dir,
    )


def build_cifar100_source(data_dir: Path) -> ImageDatasetSource:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )
    return ImageDatasetSource(
        name="cifar100",
        class_ids=tuple(range(100)),
        dataset_factory=_vision_factory(
            lambda split: datasets.CIFAR100(
                root=data_dir,
                train=split is DatasetSplit.TRAIN,
                transform=transform,
                download=True,
            )
        ),
        root=data_dir,
    )


def _vision_factory(
    factory: Callable[[DatasetSplit], Dataset[Any]],
) -> Callable[[DatasetSplit], Dataset[Any]]:
    return factory
