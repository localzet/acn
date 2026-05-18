from collections.abc import Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from acn.continual import ReplayBuffer, ReplayBufferConfig


class ReplayToyDataset(Dataset[tuple[Tensor, int]]):
    def __init__(self, targets: Sequence[int]) -> None:
        self._targets = tuple(targets)

    def __len__(self) -> int:
        return len(self._targets)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        target = self._targets[index]
        return torch.full((1, 2, 2), float(target)), target


def test_replay_buffer_enforces_capacity_reproducibly() -> None:
    dataset = ReplayToyDataset([0, 0, 1, 1, 2, 2, 3, 3])
    first = ReplayBuffer(ReplayBufferConfig(capacity=3, seed=7))
    second = ReplayBuffer(ReplayBufferConfig(capacity=3, seed=7))

    first.add_dataset(dataset)
    second.add_dataset(dataset)

    assert len(first) == 3
    assert first.class_counts() == second.class_counts()


def test_replay_dataset_returns_cloned_tensors() -> None:
    buffer = ReplayBuffer(ReplayBufferConfig(capacity=4, seed=1))
    buffer.add_dataset(ReplayToyDataset([1, 1]))
    dataset = buffer.as_dataset()

    image, target = dataset[0]
    image.fill_(99)
    restored_image, restored_target = dataset[0]

    assert target == restored_target == 1
    assert restored_image.max().item() == 1.0
