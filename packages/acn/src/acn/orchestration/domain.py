from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

from acn.controller.domain import MetricPoint
from acn.versioning.domain import Metadata


class ExperimentStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StageExecutionStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class ExperimentRecord:
    id: str
    name: str
    branch_name: str
    status: ExperimentStatus
    current_stage_id: str | None = None
    current_commit_id: str | None = None
    best_commit_id: str | None = None
    metadata: Metadata = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True, slots=True)
class StageExecutionRecord:
    id: str
    experiment_id: str
    stage_id: str
    status: StageExecutionStatus
    commit_id: str | None = None
    metrics: Metadata = field(default_factory=dict)
    started_at: datetime | None = None
    completed_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class StageTrainingResult:
    checkpoint_uri: str
    checkpoint_hash: str
    metrics: tuple[MetricPoint, ...]
    size_bytes: int | None = None
    metadata: Metadata = field(default_factory=dict)
