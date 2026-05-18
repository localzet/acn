"""add experiment commit integrity

Revision ID: 20260518_0004
Revises: 20260517_0003
Create Date: 2026-05-18 00:00:00
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260518_0004"
down_revision: str | None = "20260517_0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index(
        op.f("ix_experiments_current_commit_id"),
        "experiments",
        ["current_commit_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_experiments_best_commit_id"),
        "experiments",
        ["best_commit_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_experiment_stage_executions_commit_id"),
        "experiment_stage_executions",
        ["commit_id"],
        unique=False,
    )
    op.create_foreign_key(
        "fk_experiments_current_commit_id_training_commits",
        "experiments",
        "training_commits",
        ["current_commit_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        "fk_experiments_best_commit_id_training_commits",
        "experiments",
        "training_commits",
        ["best_commit_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        "fk_experiment_stage_executions_commit_id_training_commits",
        "experiment_stage_executions",
        "training_commits",
        ["commit_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        "fk_experiment_stage_executions_experiment_id_experiments",
        "experiment_stage_executions",
        "experiments",
        ["experiment_id"],
        ["id"],
        ondelete="CASCADE",
    )


def downgrade() -> None:
    op.drop_constraint(
        "fk_experiment_stage_executions_experiment_id_experiments",
        "experiment_stage_executions",
        type_="foreignkey",
    )
    op.drop_constraint(
        "fk_experiment_stage_executions_commit_id_training_commits",
        "experiment_stage_executions",
        type_="foreignkey",
    )
    op.drop_constraint(
        "fk_experiments_best_commit_id_training_commits",
        "experiments",
        type_="foreignkey",
    )
    op.drop_constraint(
        "fk_experiments_current_commit_id_training_commits",
        "experiments",
        type_="foreignkey",
    )
    op.drop_index(
        op.f("ix_experiment_stage_executions_commit_id"),
        table_name="experiment_stage_executions",
    )
    op.drop_index(op.f("ix_experiments_best_commit_id"), table_name="experiments")
    op.drop_index(op.f("ix_experiments_current_commit_id"), table_name="experiments")
