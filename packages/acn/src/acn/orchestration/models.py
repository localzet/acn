from datetime import UTC, datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from acn.versioning.models import Base, CommitModel, _json_type


def _now() -> datetime:
    return datetime.now(UTC)


class ExperimentModel(Base):
    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    branch_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    current_stage_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    current_commit_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("training_commits.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    best_commit_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("training_commits.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    experiment_metadata: Mapped[dict[str, Any]] = mapped_column(
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
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_now,
        onupdate=_now,
    )
    current_commit: Mapped[CommitModel | None] = relationship(
        foreign_keys=[current_commit_id],
    )
    best_commit: Mapped[CommitModel | None] = relationship(
        foreign_keys=[best_commit_id],
    )


class StageExecutionModel(Base):
    __tablename__ = "experiment_stage_executions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    experiment_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("experiments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    stage_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    commit_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("training_commits.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    metrics: Mapped[dict[str, Any]] = mapped_column(_json_type(), nullable=False, default=dict)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    commit: Mapped[CommitModel | None] = relationship()
