import pytest
from sqlalchemy.orm import Session

from acn.versioning.exceptions import (
    BranchAlreadyExistsError,
    ImmutableCheckpointError,
    InvalidRollbackTargetError,
)
from acn.versioning.models import StableCheckpointModel
from acn.versioning.repository import SqlAlchemyTrainingVersionRepository


def test_create_commits_and_branch_history(session: Session) -> None:
    repository = SqlAlchemyTrainingVersionRepository(session)
    first_checkpoint = repository.create_checkpoint(
        uri="s3://mlflow/run-1/model.pt",
        content_hash="sha256:first",
        metadata={"dataset": "fashion-mnist"},
    )
    second_checkpoint = repository.create_checkpoint(
        uri="s3://mlflow/run-2/model.pt",
        content_hash="sha256:second",
    )
    branch = repository.create_branch(name="main")

    first_commit = repository.create_commit(
        branch_name=branch.name,
        checkpoint_id=first_checkpoint.id,
        message="baseline",
        metrics={"accuracy": 0.82},
    )
    second_commit = repository.create_commit(
        branch_name=branch.name,
        checkpoint_id=second_checkpoint.id,
        message="improve classifier",
    )

    history = repository.list_branch_history("main")
    stored_commit = repository.get_commit(first_commit.id)

    assert [commit.id for commit in history] == [second_commit.id, first_commit.id]
    assert history[0].parent_commit_id == first_commit.id
    assert stored_commit.metrics == {"accuracy": 0.82}
    assert repository.get_branch("main").head_commit_id == second_commit.id


def test_create_branch_from_existing_commit(session: Session) -> None:
    repository = SqlAlchemyTrainingVersionRepository(session)
    checkpoint = repository.create_checkpoint(uri="s3://mlflow/base.pt", content_hash="sha256:base")
    repository.create_branch(name="main")
    base_commit = repository.create_commit(
        branch_name="main",
        checkpoint_id=checkpoint.id,
        message="base",
    )

    branch = repository.create_branch(name="experiment", base_commit_id=base_commit.id)

    assert branch.base_commit_id == base_commit.id
    assert branch.head_commit_id == base_commit.id
    assert repository.list_branch_history("experiment")[0].id == base_commit.id


def test_commit_graph_contains_parent_child_edges(session: Session) -> None:
    repository = SqlAlchemyTrainingVersionRepository(session)
    first_checkpoint = repository.create_checkpoint(
        uri="s3://mlflow/a.pt",
        content_hash="sha256:a",
    )
    second_checkpoint = repository.create_checkpoint(
        uri="s3://mlflow/b.pt",
        content_hash="sha256:b",
    )
    repository.create_branch(name="main")
    parent = repository.create_commit(
        branch_name="main",
        checkpoint_id=first_checkpoint.id,
        message="parent",
    )
    child = repository.create_commit(
        branch_name="main",
        checkpoint_id=second_checkpoint.id,
        message="child",
    )

    graph = repository.get_commit_graph()

    assert {node.id for node in graph.nodes} == {parent.id, child.id}
    assert graph.edges == ((parent.id, child.id),) or {
        (edge.parent_id, edge.child_id) for edge in graph.edges
    } == {(parent.id, child.id)}


def test_rollback_moves_branch_head_to_reachable_commit(session: Session) -> None:
    repository = SqlAlchemyTrainingVersionRepository(session)
    first_checkpoint = repository.create_checkpoint(
        uri="s3://mlflow/a.pt",
        content_hash="sha256:a",
    )
    second_checkpoint = repository.create_checkpoint(
        uri="s3://mlflow/b.pt",
        content_hash="sha256:b",
    )
    repository.create_branch(name="main")
    first_commit = repository.create_commit(
        branch_name="main",
        checkpoint_id=first_checkpoint.id,
        message="first",
    )
    repository.create_commit(
        branch_name="main",
        checkpoint_id=second_checkpoint.id,
        message="second",
    )

    branch = repository.rollback_branch(branch_name="main", target_commit_id=first_commit.id)

    assert branch.head_commit_id == first_commit.id
    assert [commit.id for commit in repository.list_branch_history("main")] == [first_commit.id]


def test_rollback_rejects_unreachable_commit(session: Session) -> None:
    repository = SqlAlchemyTrainingVersionRepository(session)
    main_checkpoint = repository.create_checkpoint(
        uri="s3://mlflow/main.pt",
        content_hash="sha256:main",
    )
    other_checkpoint = repository.create_checkpoint(
        uri="s3://mlflow/other.pt",
        content_hash="sha256:other",
    )
    repository.create_branch(name="main")
    repository.create_branch(name="other")
    repository.create_commit(
        branch_name="main",
        checkpoint_id=main_checkpoint.id,
        message="main",
    )
    other_commit = repository.create_commit(
        branch_name="other",
        checkpoint_id=other_checkpoint.id,
        message="other",
    )

    with pytest.raises(InvalidRollbackTargetError):
        repository.rollback_branch(branch_name="main", target_commit_id=other_commit.id)


def test_branch_names_are_unique(session: Session) -> None:
    repository = SqlAlchemyTrainingVersionRepository(session)
    repository.create_branch(name="main")

    with pytest.raises(BranchAlreadyExistsError):
        repository.create_branch(name="main")


def test_stable_checkpoint_cannot_be_mutated(session: Session) -> None:
    repository = SqlAlchemyTrainingVersionRepository(session)
    checkpoint = repository.create_checkpoint(
        uri="s3://mlflow/model.pt",
        content_hash="sha256:model",
    )
    model = session.get_one(StableCheckpointModel, checkpoint.id)

    model.uri = "s3://mlflow/changed.pt"

    with pytest.raises(ImmutableCheckpointError):
        session.flush()


def test_stable_checkpoint_cannot_be_deleted(session: Session) -> None:
    repository = SqlAlchemyTrainingVersionRepository(session)
    checkpoint = repository.create_checkpoint(
        uri="s3://mlflow/model.pt",
        content_hash="sha256:model",
    )
    model = session.get_one(StableCheckpointModel, checkpoint.id)

    session.delete(model)

    with pytest.raises(ImmutableCheckpointError):
        session.flush()
