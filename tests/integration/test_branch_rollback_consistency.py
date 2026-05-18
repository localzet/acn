import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from acn.versioning.exceptions import InvalidRollbackTargetError
from acn.versioning.models import Base
from acn.versioning.repository import SqlAlchemyTrainingVersionRepository


def test_branch_history_remains_consistent_after_branch_and_rollback() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)

    with factory() as session:
        repository = SqlAlchemyTrainingVersionRepository(session)
        first_checkpoint = repository.create_checkpoint(uri="memory://a", content_hash="sha:a")
        second_checkpoint = repository.create_checkpoint(uri="memory://b", content_hash="sha:b")
        third_checkpoint = repository.create_checkpoint(uri="memory://c", content_hash="sha:c")
        repository.create_branch(name="main")
        first = repository.create_commit(
            branch_name="main",
            checkpoint_id=first_checkpoint.id,
            message="first",
            commit_id="cmt_first",
        )
        second = repository.create_commit(
            branch_name="main",
            checkpoint_id=second_checkpoint.id,
            message="second",
            commit_id="cmt_second",
        )
        repository.create_branch(name="experiment", base_commit_id=second.id)
        experimental = repository.create_commit(
            branch_name="experiment",
            checkpoint_id=third_checkpoint.id,
            message="experiment",
            commit_id="cmt_experiment",
        )

        rolled_back = repository.rollback_branch(
            branch_name="main",
            target_commit_id=first.id,
        )
        graph = repository.get_commit_graph()

        assert rolled_back.head_commit_id == first.id
        assert [commit.id for commit in repository.list_branch_history("main")] == [first.id]
        assert [commit.id for commit in repository.list_branch_history("experiment")] == [
            experimental.id,
            second.id,
            first.id,
        ]
        assert {(edge.parent_id, edge.child_id) for edge in graph.edges} == {
            (first.id, second.id),
            (second.id, experimental.id),
        }


def test_rollback_cannot_cross_to_unreachable_experimental_head() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)

    with factory() as session:
        repository = SqlAlchemyTrainingVersionRepository(session)
        main_checkpoint = repository.create_checkpoint(uri="memory://main", content_hash="sha:main")
        exp_checkpoint = repository.create_checkpoint(uri="memory://exp", content_hash="sha:exp")
        repository.create_branch(name="main")
        main_commit = repository.create_commit(
            branch_name="main",
            checkpoint_id=main_checkpoint.id,
            message="main",
        )
        repository.create_branch(name="experiment", base_commit_id=main_commit.id)
        exp_commit = repository.create_commit(
            branch_name="experiment",
            checkpoint_id=exp_checkpoint.id,
            message="experiment",
        )

        with pytest.raises(InvalidRollbackTargetError):
            repository.rollback_branch(branch_name="main", target_commit_id=exp_commit.id)
