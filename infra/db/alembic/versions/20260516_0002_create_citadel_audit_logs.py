"""create citadel audit logs

Revision ID: 20260516_0002
Revises: 20260516_0001
Create Date: 2026-05-16 00:00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260516_0002"
down_revision: str | None = "20260516_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    json_type = postgresql.JSONB(astext_type=sa.Text())

    op.create_table(
        "citadel_audit_logs",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("action", sa.String(length=64), nullable=False),
        sa.Column("actor", sa.String(length=128), nullable=False),
        sa.Column("branch_name", sa.String(length=128), nullable=False),
        sa.Column("decision", sa.String(length=32), nullable=False),
        sa.Column("reasons", json_type, nullable=False),
        sa.Column("parameters", json_type, nullable=False),
        sa.Column("override_by", sa.String(length=128), nullable=True),
        sa.Column("override_reason", sa.Text(), nullable=True),
        sa.Column("override_ticket_id", sa.String(length=128), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_citadel_audit_logs_action"), "citadel_audit_logs", ["action"])
    op.create_index(op.f("ix_citadel_audit_logs_actor"), "citadel_audit_logs", ["actor"])
    op.create_index(
        op.f("ix_citadel_audit_logs_branch_name"),
        "citadel_audit_logs",
        ["branch_name"],
    )
    op.create_index(op.f("ix_citadel_audit_logs_decision"), "citadel_audit_logs", ["decision"])


def downgrade() -> None:
    op.drop_index(op.f("ix_citadel_audit_logs_decision"), table_name="citadel_audit_logs")
    op.drop_index(op.f("ix_citadel_audit_logs_branch_name"), table_name="citadel_audit_logs")
    op.drop_index(op.f("ix_citadel_audit_logs_actor"), table_name="citadel_audit_logs")
    op.drop_index(op.f("ix_citadel_audit_logs_action"), table_name="citadel_audit_logs")
    op.drop_table("citadel_audit_logs")
