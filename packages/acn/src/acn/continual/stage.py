from dataclasses import dataclass, field
from enum import StrEnum

from acn.versioning.domain import Metadata


class DatasetSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True, slots=True)
class DatasetStageConfig:
    id: str
    source_name: str
    class_ids: tuple[int, ...]
    split: DatasetSplit = DatasetSplit.TRAIN
    domain_shift: str | None = None
    replay_ratio: float = 0.0
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DatasetStage:
    id: str
    source_name: str
    class_ids: tuple[int, ...]
    split: DatasetSplit
    introduced_class_ids: tuple[int, ...]
    domain_shift: str | None = None
    replay_ratio: float = 0.0
    metadata: Metadata = field(default_factory=dict)
