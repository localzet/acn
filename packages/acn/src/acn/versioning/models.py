from datetime import UTC, datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, event
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from acn.versioning.exceptions import ImmutableCheckpointError


class Base(DeclarativeBase):
    pass


def _now() -> datetime:
    return datetime.now(UTC)


def _json_type() -> JSON:
    return JSON().with_variant(JSONB(), "postgresql")


class StableCheckpointModel(Base):
    __tablename__ = "stable_checkpoints"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    uri: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    checkpoint_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        _json_type(),
        nullable=False,
        default=dict,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_now,
    )


class BranchModel(Base):
    __tablename__ = "training_branches"
    __table_args__ = (UniqueConstraint("name", name="uq_training_branches_name"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    head_commit_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("training_commits.id", ondelete="SET NULL", use_alter=True),
        nullable=True,
    )
    base_commit_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("training_commits.id", ondelete="SET NULL", use_alter=True),
        nullable=True,
    )
    branch_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        _json_type(),
        nullable=False,
        default=dict,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_now,
    )

    commits: Mapped[list["CommitModel"]] = relationship(
        back_populates="branch",
        foreign_keys="CommitModel.branch_id",
        cascade="all, delete-orphan",
    )
    head_commit: Mapped["CommitModel | None"] = relationship(
        foreign_keys=[head_commit_id],
        post_update=True,
    )
    base_commit: Mapped["CommitModel | None"] = relationship(
        foreign_keys=[base_commit_id],
        post_update=True,
    )


class CommitModel(Base):
    __tablename__ = "training_commits"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    branch_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("training_branches.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    checkpoint_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("stable_checkpoints.id", ondelete="RESTRICT"),
        nullable=False,
    )
    parent_commit_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("training_commits.id", ondelete="RESTRICT"),
        nullable=True,
        index=True,
    )
    message: Mapped[str] = mapped_column(Text, nullable=False)
    authored_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    metrics: Mapped[dict[str, Any]] = mapped_column(_json_type(), nullable=False, default=dict)
    commit_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        _json_type(),
        nullable=False,
        default=dict,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_now,
    )

    branch: Mapped[BranchModel] = relationship(
        back_populates="commits",
        foreign_keys=[branch_id],
    )
    checkpoint: Mapped[StableCheckpointModel] = relationship()
    parent: Mapped["CommitModel | None"] = relationship(
        remote_side=lambda: [CommitModel.id],
        foreign_keys=[parent_commit_id],
    )


@event.listens_for(StableCheckpointModel, "before_update")
@event.listens_for(StableCheckpointModel, "before_delete")
def _prevent_checkpoint_mutation(
    _mapper: object,
    _connection: object,
    _target: StableCheckpointModel,
) -> None:
    msg = "Stable checkpoints are immutable."
    raise ImmutableCheckpointError(msg)
