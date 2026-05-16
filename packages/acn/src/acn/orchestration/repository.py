from datetime import UTC, datetime
from typing import Protocol
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from acn.orchestration.domain import (
    ExperimentRecord,
    ExperimentStatus,
    StageExecutionRecord,
    StageExecutionStatus,
)
from acn.orchestration.models import ExperimentModel, StageExecutionModel
from acn.versioning.domain import Metadata


class ExperimentStateRepository(Protocol):
    def create_experiment(
        self,
        *,
        name: str,
        branch_name: str,
        metadata: Metadata | None = None,
        experiment_id: str | None = None,
    ) -> ExperimentRecord: ...

    def get_experiment(self, experiment_id: str) -> ExperimentRecord: ...

    def update_experiment(
        self,
        experiment_id: str,
        *,
        status: ExperimentStatus | None = None,
        current_stage_id: str | None = None,
        current_commit_id: str | None = None,
        best_commit_id: str | None = None,
    ) -> ExperimentRecord: ...

    def create_stage_execution(
        self,
        *,
        experiment_id: str,
        stage_id: str,
        status: StageExecutionStatus = StageExecutionStatus.PENDING,
        execution_id: str | None = None,
    ) -> StageExecutionRecord: ...

    def update_stage_execution(
        self,
        execution_id: str,
        *,
        status: StageExecutionStatus,
        commit_id: str | None = None,
        metrics: Metadata | None = None,
    ) -> StageExecutionRecord: ...

    def list_stage_executions(self, experiment_id: str) -> tuple[StageExecutionRecord, ...]: ...


class InMemoryExperimentStateRepository:
    def __init__(self) -> None:
        self._experiments: dict[str, ExperimentRecord] = {}
        self._stage_executions: dict[str, StageExecutionRecord] = {}

    def create_experiment(
        self,
        *,
        name: str,
        branch_name: str,
        metadata: Metadata | None = None,
        experiment_id: str | None = None,
    ) -> ExperimentRecord:
        record = ExperimentRecord(
            id=experiment_id or _new_id("exp"),
            name=name,
            branch_name=branch_name,
            status=ExperimentStatus.CREATED,
            metadata=metadata or {},
        )
        self._experiments[record.id] = record
        return record

    def get_experiment(self, experiment_id: str) -> ExperimentRecord:
        return self._experiments[experiment_id]

    def update_experiment(
        self,
        experiment_id: str,
        *,
        status: ExperimentStatus | None = None,
        current_stage_id: str | None = None,
        current_commit_id: str | None = None,
        best_commit_id: str | None = None,
    ) -> ExperimentRecord:
        current = self._experiments[experiment_id]
        updated = ExperimentRecord(
            id=current.id,
            name=current.name,
            branch_name=current.branch_name,
            status=status or current.status,
            current_stage_id=(
                current_stage_id if current_stage_id is not None else current.current_stage_id
            ),
            current_commit_id=(
                current_commit_id if current_commit_id is not None else current.current_commit_id
            ),
            best_commit_id=best_commit_id if best_commit_id is not None else current.best_commit_id,
            metadata=current.metadata,
            created_at=current.created_at,
            updated_at=datetime.now(UTC),
        )
        self._experiments[experiment_id] = updated
        return updated

    def create_stage_execution(
        self,
        *,
        experiment_id: str,
        stage_id: str,
        status: StageExecutionStatus = StageExecutionStatus.PENDING,
        execution_id: str | None = None,
    ) -> StageExecutionRecord:
        record = StageExecutionRecord(
            id=execution_id or _new_id("stage"),
            experiment_id=experiment_id,
            stage_id=stage_id,
            status=status,
            started_at=datetime.now(UTC) if status is StageExecutionStatus.RUNNING else None,
        )
        self._stage_executions[record.id] = record
        return record

    def update_stage_execution(
        self,
        execution_id: str,
        *,
        status: StageExecutionStatus,
        commit_id: str | None = None,
        metrics: Metadata | None = None,
    ) -> StageExecutionRecord:
        current = self._stage_executions[execution_id]
        completed_at = datetime.now(UTC) if status is StageExecutionStatus.COMPLETED else None
        updated = StageExecutionRecord(
            id=current.id,
            experiment_id=current.experiment_id,
            stage_id=current.stage_id,
            status=status,
            commit_id=commit_id if commit_id is not None else current.commit_id,
            metrics=metrics or current.metrics,
            started_at=current.started_at,
            completed_at=completed_at or current.completed_at,
        )
        self._stage_executions[execution_id] = updated
        return updated

    def list_stage_executions(self, experiment_id: str) -> tuple[StageExecutionRecord, ...]:
        return tuple(
            record
            for record in self._stage_executions.values()
            if record.experiment_id == experiment_id
        )


class SqlAlchemyExperimentStateRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create_experiment(
        self,
        *,
        name: str,
        branch_name: str,
        metadata: Metadata | None = None,
        experiment_id: str | None = None,
    ) -> ExperimentRecord:
        model = ExperimentModel(
            id=experiment_id or _new_id("exp"),
            name=name,
            branch_name=branch_name,
            status=ExperimentStatus.CREATED.value,
            experiment_metadata=metadata or {},
        )
        self._session.add(model)
        self._session.flush()
        return _experiment_record(model)

    def get_experiment(self, experiment_id: str) -> ExperimentRecord:
        return _experiment_record(self._session.get_one(ExperimentModel, experiment_id))

    def update_experiment(
        self,
        experiment_id: str,
        *,
        status: ExperimentStatus | None = None,
        current_stage_id: str | None = None,
        current_commit_id: str | None = None,
        best_commit_id: str | None = None,
    ) -> ExperimentRecord:
        model = self._session.get_one(ExperimentModel, experiment_id)
        if status is not None:
            model.status = status.value
        if current_stage_id is not None:
            model.current_stage_id = current_stage_id
        if current_commit_id is not None:
            model.current_commit_id = current_commit_id
        if best_commit_id is not None:
            model.best_commit_id = best_commit_id
        self._session.flush()
        return _experiment_record(model)

    def create_stage_execution(
        self,
        *,
        experiment_id: str,
        stage_id: str,
        status: StageExecutionStatus = StageExecutionStatus.PENDING,
        execution_id: str | None = None,
    ) -> StageExecutionRecord:
        model = StageExecutionModel(
            id=execution_id or _new_id("stage"),
            experiment_id=experiment_id,
            stage_id=stage_id,
            status=status.value,
            started_at=datetime.now(UTC) if status is StageExecutionStatus.RUNNING else None,
        )
        self._session.add(model)
        self._session.flush()
        return _stage_execution_record(model)

    def update_stage_execution(
        self,
        execution_id: str,
        *,
        status: StageExecutionStatus,
        commit_id: str | None = None,
        metrics: Metadata | None = None,
    ) -> StageExecutionRecord:
        model = self._session.get_one(StageExecutionModel, execution_id)
        model.status = status.value
        if commit_id is not None:
            model.commit_id = commit_id
        if metrics is not None:
            model.metrics = metrics
        if status is StageExecutionStatus.COMPLETED:
            model.completed_at = datetime.now(UTC)
        self._session.flush()
        return _stage_execution_record(model)

    def list_stage_executions(self, experiment_id: str) -> tuple[StageExecutionRecord, ...]:
        records = self._session.scalars(
            select(StageExecutionModel).where(StageExecutionModel.experiment_id == experiment_id)
        ).all()
        return tuple(_stage_execution_record(record) for record in records)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def _experiment_record(model: ExperimentModel) -> ExperimentRecord:
    return ExperimentRecord(
        id=model.id,
        name=model.name,
        branch_name=model.branch_name,
        status=ExperimentStatus(model.status),
        current_stage_id=model.current_stage_id,
        current_commit_id=model.current_commit_id,
        best_commit_id=model.best_commit_id,
        metadata=dict(model.experiment_metadata),
        created_at=model.created_at,
        updated_at=model.updated_at,
    )


def _stage_execution_record(model: StageExecutionModel) -> StageExecutionRecord:
    return StageExecutionRecord(
        id=model.id,
        experiment_id=model.experiment_id,
        stage_id=model.stage_id,
        status=StageExecutionStatus(model.status),
        commit_id=model.commit_id,
        metrics=dict(model.metrics),
        started_at=model.started_at,
        completed_at=model.completed_at,
    )
