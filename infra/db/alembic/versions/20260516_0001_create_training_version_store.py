"""create training version store

Revision ID: 20260516_0001
Revises:
Create Date: 2026-05-16 00:00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260516_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    json_type = postgresql.JSONB(astext_type=sa.Text())

    op.create_table(
        "stable_checkpoints",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("uri", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(length=128), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=True),
        sa.Column("metadata", json_type, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("content_hash"),
        sa.UniqueConstraint("uri"),
    )
    op.create_table(
        "training_branches",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("head_commit_id", sa.String(length=64), nullable=True),
        sa.Column("base_commit_id", sa.String(length=64), nullable=True),
        sa.Column("metadata", json_type, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", name="uq_training_branches_name"),
    )
    op.create_table(
        "training_commits",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("branch_id", sa.String(length=64), nullable=False),
        sa.Column("checkpoint_id", sa.String(length=64), nullable=False),
        sa.Column("parent_commit_id", sa.String(length=64), nullable=True),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("authored_by", sa.String(length=128), nullable=True),
        sa.Column("metrics", json_type, nullable=False),
        sa.Column("metadata", json_type, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["branch_id"], ["training_branches.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["checkpoint_id"], ["stable_checkpoints.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(
            ["parent_commit_id"],
            ["training_commits.id"],
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_training_commits_branch_id"),
        "training_commits",
        ["branch_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_training_commits_parent_commit_id"),
        "training_commits",
        ["parent_commit_id"],
        unique=False,
    )
    op.create_foreign_key(
        "fk_training_branches_head_commit_id_training_commits",
        "training_branches",
        "training_commits",
        ["head_commit_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        "fk_training_branches_base_commit_id_training_commits",
        "training_branches",
        "training_commits",
        ["base_commit_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint(
        "fk_training_branches_base_commit_id_training_commits",
        "training_branches",
        type_="foreignkey",
    )
    op.drop_constraint(
        "fk_training_branches_head_commit_id_training_commits",
        "training_branches",
        type_="foreignkey",
    )
    op.drop_index(op.f("ix_training_commits_parent_commit_id"), table_name="training_commits")
    op.drop_index(op.f("ix_training_commits_branch_id"), table_name="training_commits")
    op.drop_table("training_commits")
    op.drop_table("training_branches")
    op.drop_table("stable_checkpoints")
