from datetime import UTC, datetime
from typing import Any

from sqlalchemy import DateTime, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from acn.versioning.models import Base, _json_type


def _now() -> datetime:
    return datetime.now(UTC)


class CitadelAuditLogModel(Base):
    __tablename__ = "citadel_audit_logs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    action: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    actor: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    branch_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    decision: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    reasons: Mapped[list[str]] = mapped_column(_json_type(), nullable=False, default=list)
    parameters: Mapped[dict[str, Any]] = mapped_column(_json_type(), nullable=False, default=dict)
    override_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    override_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    override_ticket_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_now,
    )
