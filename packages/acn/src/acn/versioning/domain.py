from dataclasses import dataclass, field
from datetime import UTC, datetime

type MetadataValue = (
    str | int | float | bool | None | list["MetadataValue"] | dict[str, "MetadataValue"]
)
type Metadata = dict[str, MetadataValue]


@dataclass(frozen=True, slots=True)
class StableCheckpointRecord:
    id: str
    uri: str
    content_hash: str
    size_bytes: int | None = None
    metadata: Metadata = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True, slots=True)
class BranchRecord:
    id: str
    name: str
    head_commit_id: str | None = None
    base_commit_id: str | None = None
    metadata: Metadata = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True, slots=True)
class CommitRecord:
    id: str
    branch_id: str
    checkpoint_id: str
    message: str
    parent_commit_id: str | None = None
    authored_by: str | None = None
    metrics: Metadata = field(default_factory=dict)
    metadata: Metadata = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True, slots=True)
class CommitGraphNode:
    id: str
    branch_id: str
    checkpoint_id: str
    message: str
    created_at: datetime
    metadata: Metadata = field(default_factory=dict)
    metrics: Metadata = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CommitGraphEdge:
    parent_id: str
    child_id: str


@dataclass(frozen=True, slots=True)
class CommitGraph:
    nodes: tuple[CommitGraphNode, ...]
    edges: tuple[CommitGraphEdge, ...]
