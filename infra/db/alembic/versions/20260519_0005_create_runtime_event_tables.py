"""create runtime event tables

Revision ID: 20260519_0005
Revises: 20260518_0004
Create Date: 2026-05-19 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260519_0005"
down_revision: str | None = "20260518_0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "controller_decisions",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("experiment_id", sa.String(length=64), nullable=False),
        sa.Column("action", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("commit_id", sa.String(length=64), nullable=True),
        sa.Column("mlflow_run_id", sa.String(length=128), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["commit_id"], ["training_commits.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["experiment_id"], ["experiments.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_controller_decisions_action", "controller_decisions", ["action"])
    op.create_index("ix_controller_decisions_commit_id", "controller_decisions", ["commit_id"])
    op.create_index(
        "ix_controller_decisions_experiment_id",
        "controller_decisions",
        ["experiment_id"],
    )
    op.create_index("ix_controller_decisions_status", "controller_decisions", ["status"])

    op.create_table(
        "rollback_events",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("experiment_id", sa.String(length=64), nullable=False),
        sa.Column("branch_name", sa.String(length=128), nullable=False),
        sa.Column("from_commit_id", sa.String(length=64), nullable=True),
        sa.Column("to_commit_id", sa.String(length=64), nullable=True),
        sa.Column("artifact_uri", sa.Text(), nullable=True),
        sa.Column("mlflow_run_id", sa.String(length=128), nullable=True),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["experiment_id"], ["experiments.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["from_commit_id"], ["training_commits.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["to_commit_id"], ["training_commits.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_rollback_events_branch_name", "rollback_events", ["branch_name"])
    op.create_index("ix_rollback_events_experiment_id", "rollback_events", ["experiment_id"])


def downgrade() -> None:
    op.drop_index("ix_rollback_events_experiment_id", table_name="rollback_events")
    op.drop_index("ix_rollback_events_branch_name", table_name="rollback_events")
    op.drop_table("rollback_events")
    op.drop_index("ix_controller_decisions_status", table_name="controller_decisions")
    op.drop_index("ix_controller_decisions_experiment_id", table_name="controller_decisions")
    op.drop_index("ix_controller_decisions_commit_id", table_name="controller_decisions")
    op.drop_index("ix_controller_decisions_action", table_name="controller_decisions")
    op.drop_table("controller_decisions")
