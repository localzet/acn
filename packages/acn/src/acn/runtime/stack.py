import io
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from uuid import uuid4

import torch
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from acn.artifacts import ArtifactReference, CheckpointArtifactPayload, MinIOArtifactStore
from acn.config.settings import Settings
from acn.orchestration.domain import ExperimentStatus
from acn.orchestration.models import ExperimentModel
from acn.runtime.models import ControllerDecisionModel, RollbackEventModel
from acn.versioning.domain import CommitRecord, Metadata
from acn.versioning.models import Base, BranchModel
from acn.versioning.repository import SqlAlchemyTrainingVersionRepository


@dataclass(frozen=True, slots=True)
class RuntimeStatusItem:
    name: str
    connected: bool
    message: str


@dataclass(frozen=True, slots=True)
class RuntimeStatus:
    postgres: RuntimeStatusItem
    mlflow: RuntimeStatusItem
    minio: RuntimeStatusItem
    artifact_storage: RuntimeStatusItem


class RuntimeStack:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._engine = create_engine(settings.database_url, future=True)
        self._session_factory = sessionmaker(bind=self._engine, expire_on_commit=False, future=True)
        self._artifact_store: MinIOArtifactStore | None = None

    @property
    def session_factory(self) -> sessionmaker[Session]:
        return self._session_factory

    @property
    def artifact_store(self) -> MinIOArtifactStore:
        if self._artifact_store is None:
            self._artifact_store = MinIOArtifactStore(
                endpoint_url=self._settings.minio_endpoint,
                access_key=self._settings.minio_access_key,
                secret_key=self._settings.minio_secret_key,
                bucket=self._settings.minio_artifact_bucket,
                region=self._settings.minio_region,
            )
        return self._artifact_store

    def initialize(self) -> RuntimeStatus:
        postgres = self._check_postgres()
        if postgres.connected:
            Base.metadata.create_all(self._engine)
        return RuntimeStatus(
            postgres=postgres,
            mlflow=self._check_mlflow(),
            minio=self._check_minio(),
            artifact_storage=self._check_artifact_storage(),
        )

    def start_mlflow_run(self, *, run_name: str, params: Metadata) -> str | None:
        try:
            import mlflow

            mlflow.set_tracking_uri(self._settings.mlflow_tracking_uri)
            mlflow.set_experiment(self._settings.mlflow_experiment_name)
            active = mlflow.start_run(run_name=run_name)
            mlflow.log_params(params)
            return cast(str, active.info.run_id)
        except Exception:
            return None

    def end_mlflow_run(self) -> None:
        try:
            import mlflow

            mlflow.end_run()
        except Exception:
            return

    def log_mlflow_metric(self, key: str, value: float, *, step: int) -> None:
        try:
            import mlflow

            mlflow.log_metric(key, value, step=step)
        except Exception:
            return

    def log_mlflow_text(self, name: str, text_value: str) -> None:
        try:
            import mlflow

            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / name
                path.write_text(text_value, encoding="utf-8")
                mlflow.log_artifact(str(path))
        except Exception:
            return

    def save_checkpoint(
        self,
        *,
        name: str,
        payload: CheckpointArtifactPayload,
    ) -> ArtifactReference | None:
        try:
            reference = self.artifact_store.save_checkpoint(name=name, payload=payload)
            self._log_checkpoint_to_mlflow(name=name, payload=payload)
            return reference
        except Exception:
            return None

    def load_checkpoint(
        self,
        artifact_uri: str,
        *,
        expected_checksum: str | None = None,
        map_location: str | torch.device = "cpu",
    ) -> CheckpointArtifactPayload:
        return self.artifact_store.load_checkpoint(
            artifact_uri,
            expected_checksum=expected_checksum,
            map_location=map_location,
        )

    def ensure_visual_experiment(self, *, experiment_id: str, run_id: str | None) -> None:
        with self._session_factory() as session, session.begin():
            if session.get(BranchModel, "br_visual_demo") is None:
                repository = SqlAlchemyTrainingVersionRepository(session)
                with suppress(Exception):
                    repository.create_branch(
                        name="visual-demo",
                        branch_id="br_visual_demo",
                        metadata={"runtime": "postgres"},
                    )
            if session.get(ExperimentModel, experiment_id) is None:
                session.add(
                    ExperimentModel(
                        id=experiment_id,
                        name="visual-adaptive-demo",
                        branch_name="visual-demo",
                        status=ExperimentStatus.RUNNING.value,
                        experiment_metadata={"mlflow_run_id": run_id},
                    )
                )

    def persist_checkpoint_commit(
        self,
        *,
        experiment_id: str,
        checkpoint_id: str,
        artifact: ArtifactReference | None,
        metrics: Metadata,
        mlflow_run_id: str | None,
    ) -> CommitRecord | None:
        if artifact is None:
            return None
        with self._session_factory() as session, session.begin():
            repository = SqlAlchemyTrainingVersionRepository(session)
            checkpoint = repository.create_checkpoint(
                checkpoint_id=f"chk_{checkpoint_id}",
                uri=artifact.uri,
                content_hash=artifact.checksum,
                size_bytes=artifact.size_bytes,
                metadata={
                    "experiment_id": experiment_id,
                    "storage": "minio",
                    "mlflow_run_id": mlflow_run_id,
                },
            )
            commit = repository.create_commit(
                branch_name="visual-demo",
                checkpoint_id=checkpoint.id,
                message=f"visual-demo:{checkpoint_id}",
                authored_by="visual-demo",
                metrics=metrics,
                metadata={
                    "artifact_uri": artifact.uri,
                    "artifact_size_bytes": artifact.size_bytes,
                    "storage": "minio",
                    "mlflow_run_id": mlflow_run_id,
                },
                commit_id=checkpoint_id,
            )
            experiment = session.get(ExperimentModel, experiment_id)
            if experiment is not None:
                experiment.current_stage_id = str(metrics.get("stage", "adaptive-training"))
                experiment.current_commit_id = commit.id
                if experiment.best_commit_id is None or bool(metrics.get("stable", False)):
                    experiment.best_commit_id = commit.id
            return commit

    def persist_decision(
        self,
        *,
        experiment_id: str,
        decision_id: str,
        action: str,
        status: str,
        reason: str,
        commit_id: str | None,
        mlflow_run_id: str | None,
        metadata: Metadata | None = None,
    ) -> None:
        with self._session_factory() as session, session.begin():
            session.merge(
                ControllerDecisionModel(
                    id=decision_id,
                    experiment_id=experiment_id,
                    action=action,
                    status=status,
                    reason=reason,
                    commit_id=commit_id,
                    mlflow_run_id=mlflow_run_id,
                    decision_metadata=metadata or {},
                )
            )

    def persist_rollback(
        self,
        *,
        experiment_id: str,
        from_commit_id: str | None,
        to_commit_id: str | None,
        artifact_uri: str | None,
        reason: str,
        mlflow_run_id: str | None,
    ) -> None:
        with self._session_factory() as session, session.begin():
            session.add(
                RollbackEventModel(
                    id=f"rb_{uuid4().hex}",
                    experiment_id=experiment_id,
                    branch_name="visual-demo",
                    from_commit_id=from_commit_id,
                    to_commit_id=to_commit_id,
                    artifact_uri=artifact_uri,
                    mlflow_run_id=mlflow_run_id,
                    reason=reason,
                    event_metadata={"storage": "minio"},
                )
            )

    def artifact_browser(self) -> list[dict[str, str | int | float | bool | None]]:
        with self._session_factory() as session:
            repository = SqlAlchemyTrainingVersionRepository(session)
            graph = repository.get_commit_graph()
            return [
                {
                    "commitId": node.id,
                    "checkpointId": node.checkpoint_id,
                    "artifactUri": str(node.metadata.get("artifact_uri", "")),
                    "artifactSizeBytes": _optional_int(node.metadata.get("artifact_size_bytes")),
                    "storage": str(node.metadata.get("storage", "")),
                    "mlflowRunId": _optional_str(node.metadata.get("mlflow_run_id")),
                    "validationLoss": _optional_float(node.metrics.get("validation_loss")),
                    "accuracy": _optional_float(node.metrics.get("accuracy")),
                }
                for node in graph.nodes
                if node.metadata.get("storage") == "minio"
            ]

    def _check_postgres(self) -> RuntimeStatusItem:
        try:
            with self._engine.connect() as connection:
                connection.execute(text("select 1"))
            return RuntimeStatusItem("PostgreSQL", True, "connected")
        except SQLAlchemyError as exc:
            return RuntimeStatusItem("PostgreSQL", False, str(exc.__class__.__name__))

    def _check_mlflow(self) -> RuntimeStatusItem:
        try:
            import mlflow

            mlflow.set_tracking_uri(self._settings.mlflow_tracking_uri)
            mlflow.search_experiments(max_results=1)
            return RuntimeStatusItem("MLflow", True, "connected")
        except Exception as exc:
            return RuntimeStatusItem("MLflow", False, str(exc.__class__.__name__))

    def _check_minio(self) -> RuntimeStatusItem:
        try:
            _ = self.artifact_store
            return RuntimeStatusItem("MinIO", True, "connected")
        except Exception as exc:
            return RuntimeStatusItem("MinIO", False, str(exc.__class__.__name__))

    def _check_artifact_storage(self) -> RuntimeStatusItem:
        try:
            payload: CheckpointArtifactPayload = {
                "epoch": 0,
                "global_step": 0,
                "best_validation_loss": None,
                "model_state": {},
                "optimizer_state": {},
                "scheduler_state": None,
                "scaler_state": None,
            }
            reference = self.artifact_store.save_checkpoint(
                name=".health/checkpoint.pt",
                payload=payload,
            )
            self.artifact_store.delete_checkpoint(reference.uri)
            return RuntimeStatusItem("Artifact storage", True, "writable")
        except Exception as exc:
            return RuntimeStatusItem("Artifact storage", False, str(exc.__class__.__name__))

    def _log_checkpoint_to_mlflow(self, *, name: str, payload: CheckpointArtifactPayload) -> None:
        try:
            import mlflow

            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / Path(name).name
                buffer = io.BytesIO()
                torch.save(payload, buffer)
                path.write_bytes(buffer.getvalue())
                mlflow.log_artifact(str(path), artifact_path="checkpoints")
        except Exception:
            return


def _optional_float(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def _optional_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    return None


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None
