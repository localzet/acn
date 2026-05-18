from typing import Protocol
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from acn.versioning.domain import (
    BranchRecord,
    CommitGraph,
    CommitGraphEdge,
    CommitGraphNode,
    CommitRecord,
    Metadata,
    StableCheckpointRecord,
)
from acn.versioning.exceptions import (
    BranchAlreadyExistsError,
    BranchNotFoundError,
    CheckpointNotFoundError,
    CommitNotFoundError,
    InvalidRollbackTargetError,
)
from acn.versioning.models import BranchModel, CommitModel, StableCheckpointModel


class TrainingVersionRepository(Protocol):
    def create_checkpoint(
        self,
        *,
        uri: str,
        content_hash: str,
        size_bytes: int | None = None,
        metadata: Metadata | None = None,
        checkpoint_id: str | None = None,
    ) -> StableCheckpointRecord: ...

    def create_branch(
        self,
        *,
        name: str,
        base_commit_id: str | None = None,
        metadata: Metadata | None = None,
        branch_id: str | None = None,
    ) -> BranchRecord: ...

    def create_commit(
        self,
        *,
        branch_name: str,
        checkpoint_id: str,
        message: str,
        parent_commit_id: str | None = None,
        authored_by: str | None = None,
        metrics: Metadata | None = None,
        metadata: Metadata | None = None,
        commit_id: str | None = None,
    ) -> CommitRecord: ...

    def get_branch(self, name: str) -> BranchRecord: ...

    def get_commit(self, commit_id: str) -> CommitRecord: ...

    def get_checkpoint(self, checkpoint_id: str) -> StableCheckpointRecord: ...

    def list_branch_history(self, branch_name: str) -> tuple[CommitRecord, ...]: ...

    def rollback_branch(self, *, branch_name: str, target_commit_id: str) -> BranchRecord: ...

    def get_commit_graph(self) -> CommitGraph: ...


class SqlAlchemyTrainingVersionRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create_checkpoint(
        self,
        *,
        uri: str,
        content_hash: str,
        size_bytes: int | None = None,
        metadata: Metadata | None = None,
        checkpoint_id: str | None = None,
    ) -> StableCheckpointRecord:
        checkpoint = StableCheckpointModel(
            id=checkpoint_id or _new_id("chk"),
            uri=uri,
            content_hash=content_hash,
            size_bytes=size_bytes,
            checkpoint_metadata=metadata or {},
        )
        self._session.add(checkpoint)
        self._flush_integrity("Checkpoint URI and content hash must be unique.")
        return _checkpoint_record(checkpoint)

    def create_branch(
        self,
        *,
        name: str,
        base_commit_id: str | None = None,
        metadata: Metadata | None = None,
        branch_id: str | None = None,
    ) -> BranchRecord:
        if base_commit_id is not None:
            self._require_commit(base_commit_id)

        branch = BranchModel(
            id=branch_id or _new_id("br"),
            name=name,
            base_commit_id=base_commit_id,
            head_commit_id=base_commit_id,
            branch_metadata=metadata or {},
        )
        self._session.add(branch)
        try:
            self._session.flush()
        except IntegrityError as exc:
            self._session.rollback()
            msg = f"Branch already exists: {name}"
            raise BranchAlreadyExistsError(msg) from exc
        return _branch_record(branch)

    def create_commit(
        self,
        *,
        branch_name: str,
        checkpoint_id: str,
        message: str,
        parent_commit_id: str | None = None,
        authored_by: str | None = None,
        metrics: Metadata | None = None,
        metadata: Metadata | None = None,
        commit_id: str | None = None,
    ) -> CommitRecord:
        branch = self._require_branch(branch_name)
        self._require_checkpoint(checkpoint_id)

        resolved_parent_id = (
            parent_commit_id if parent_commit_id is not None else branch.head_commit_id
        )
        if resolved_parent_id is not None:
            self._require_commit(resolved_parent_id)

        commit = CommitModel(
            id=commit_id or _new_id("cmt"),
            branch_id=branch.id,
            checkpoint_id=checkpoint_id,
            parent_commit_id=resolved_parent_id,
            message=message,
            authored_by=authored_by,
            metrics=metrics or {},
            commit_metadata=metadata or {},
        )
        self._session.add(commit)
        self._session.flush()

        branch.head_commit_id = commit.id
        self._session.flush()
        return _commit_record(commit)

    def get_branch(self, name: str) -> BranchRecord:
        return _branch_record(self._require_branch(name))

    def get_commit(self, commit_id: str) -> CommitRecord:
        return _commit_record(self._require_commit(commit_id))

    def get_checkpoint(self, checkpoint_id: str) -> StableCheckpointRecord:
        return _checkpoint_record(self._require_checkpoint(checkpoint_id))

    def list_branch_history(self, branch_name: str) -> tuple[CommitRecord, ...]:
        branch = self._require_branch(branch_name)
        commits_by_id = self._commits_by_id()
        history: list[CommitRecord] = []
        current_id = branch.head_commit_id

        while current_id is not None:
            commit = commits_by_id.get(current_id)
            if commit is None:
                msg = f"Commit not found: {current_id}"
                raise CommitNotFoundError(msg)
            history.append(_commit_record(commit))
            current_id = commit.parent_commit_id

        return tuple(history)

    def rollback_branch(self, *, branch_name: str, target_commit_id: str) -> BranchRecord:
        branch = self._require_branch(branch_name)
        self._require_commit(target_commit_id)
        if not self._is_reachable_from_head(branch, target_commit_id):
            msg = (
                f"Commit {target_commit_id} is not reachable from branch "
                f"{branch.name} head {branch.head_commit_id}."
            )
            raise InvalidRollbackTargetError(msg)

        branch.head_commit_id = target_commit_id
        self._session.flush()
        return _branch_record(branch)

    def get_commit_graph(self) -> CommitGraph:
        commits = self._session.scalars(select(CommitModel)).all()
        nodes = tuple(
            CommitGraphNode(
                id=commit.id,
                branch_id=commit.branch_id,
                checkpoint_id=commit.checkpoint_id,
                message=commit.message,
                created_at=commit.created_at,
                metadata=dict(commit.commit_metadata),
                metrics=dict(commit.metrics),
            )
            for commit in commits
        )
        edges = tuple(
            CommitGraphEdge(parent_id=commit.parent_commit_id, child_id=commit.id)
            for commit in commits
            if commit.parent_commit_id is not None
        )
        return CommitGraph(nodes=nodes, edges=edges)

    def _require_branch(self, name: str) -> BranchModel:
        branch = self._session.scalar(select(BranchModel).where(BranchModel.name == name))
        if branch is None:
            msg = f"Branch not found: {name}"
            raise BranchNotFoundError(msg)
        return branch

    def _require_commit(self, commit_id: str) -> CommitModel:
        commit = self._session.get(CommitModel, commit_id)
        if commit is None:
            msg = f"Commit not found: {commit_id}"
            raise CommitNotFoundError(msg)
        return commit

    def _require_checkpoint(self, checkpoint_id: str) -> StableCheckpointModel:
        checkpoint = self._session.get(StableCheckpointModel, checkpoint_id)
        if checkpoint is None:
            msg = f"Checkpoint not found: {checkpoint_id}"
            raise CheckpointNotFoundError(msg)
        return checkpoint

    def _is_reachable_from_head(self, branch: BranchModel, target_commit_id: str) -> bool:
        current_id = branch.head_commit_id
        commits_by_id = self._commits_by_id()
        while current_id is not None:
            if current_id == target_commit_id:
                return True
            current_id = commits_by_id[current_id].parent_commit_id
        return False

    def _commits_by_id(self) -> dict[str, CommitModel]:
        commits = self._session.scalars(select(CommitModel)).all()
        return {commit.id: commit for commit in commits}

    def _flush_integrity(self, message: str) -> None:
        try:
            self._session.flush()
        except IntegrityError as exc:
            self._session.rollback()
            raise ValueError(message) from exc


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def _checkpoint_record(model: StableCheckpointModel) -> StableCheckpointRecord:
    return StableCheckpointRecord(
        id=model.id,
        uri=model.uri,
        content_hash=model.content_hash,
        size_bytes=model.size_bytes,
        metadata=dict(model.checkpoint_metadata),
        created_at=model.created_at,
    )


def _branch_record(model: BranchModel) -> BranchRecord:
    return BranchRecord(
        id=model.id,
        name=model.name,
        head_commit_id=model.head_commit_id,
        base_commit_id=model.base_commit_id,
        metadata=dict(model.branch_metadata),
        created_at=model.created_at,
    )


def _commit_record(model: CommitModel) -> CommitRecord:
    return CommitRecord(
        id=model.id,
        branch_id=model.branch_id,
        checkpoint_id=model.checkpoint_id,
        parent_commit_id=model.parent_commit_id,
        message=model.message,
        authored_by=model.authored_by,
        metrics=dict(model.metrics),
        metadata=dict(model.commit_metadata),
        created_at=model.created_at,
    )
