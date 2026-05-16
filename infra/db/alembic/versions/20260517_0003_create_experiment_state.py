"""create experiment state

Revision ID: 20260517_0003
Revises: 20260516_0002
Create Date: 2026-05-17 00:00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260517_0003"
down_revision: str | None = "20260516_0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    json_type = postgresql.JSONB(astext_type=sa.Text())

    op.create_table(
        "experiments",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("branch_name", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("current_stage_id", sa.String(length=128), nullable=True),
        sa.Column("current_commit_id", sa.String(length=64), nullable=True),
        sa.Column("best_commit_id", sa.String(length=64), nullable=True),
        sa.Column("metadata", json_type, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )
    op.create_index(op.f("ix_experiments_branch_name"), "experiments", ["branch_name"])
    op.create_index(op.f("ix_experiments_status"), "experiments", ["status"])

    op.create_table(
        "experiment_stage_executions",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("experiment_id", sa.String(length=64), nullable=False),
        sa.Column("stage_id", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("commit_id", sa.String(length=64), nullable=True),
        sa.Column("metrics", json_type, nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_experiment_stage_executions_experiment_id"),
        "experiment_stage_executions",
        ["experiment_id"],
    )
    op.create_index(
        op.f("ix_experiment_stage_executions_stage_id"),
        "experiment_stage_executions",
        ["stage_id"],
    )
    op.create_index(
        op.f("ix_experiment_stage_executions_status"),
        "experiment_stage_executions",
        ["status"],
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_experiment_stage_executions_status"),
        table_name="experiment_stage_executions",
    )
    op.drop_index(
        op.f("ix_experiment_stage_executions_stage_id"),
        table_name="experiment_stage_executions",
    )
    op.drop_index(
        op.f("ix_experiment_stage_executions_experiment_id"),
        table_name="experiment_stage_executions",
    )
    op.drop_table("experiment_stage_executions")
    op.drop_index(op.f("ix_experiments_status"), table_name="experiments")
    op.drop_index(op.f("ix_experiments_branch_name"), table_name="experiments")
    op.drop_table("experiments")
