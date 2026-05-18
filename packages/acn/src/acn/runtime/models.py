from datetime import UTC, datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from acn.versioning.models import Base, _json_type


def _now() -> datetime:
    return datetime.now(UTC)


class ControllerDecisionModel(Base):
    __tablename__ = "controller_decisions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    experiment_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("experiments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    action: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    commit_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("training_commits.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    mlflow_run_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    decision_metadata: Mapped[dict[str, Any]] = mapped_column(
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


class RollbackEventModel(Base):
    __tablename__ = "rollback_events"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    experiment_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("experiments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    branch_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    from_commit_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("training_commits.id", ondelete="SET NULL"),
        nullable=True,
    )
    to_commit_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("training_commits.id", ondelete="SET NULL"),
        nullable=True,
    )
    artifact_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    mlflow_run_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    event_metadata: Mapped[dict[str, Any]] = mapped_column(
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
