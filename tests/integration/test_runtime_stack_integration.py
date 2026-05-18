import os

import pytest

from acn.config.settings import Settings
from acn.runtime import RuntimeStack

pytestmark = pytest.mark.skipif(
    os.getenv("ACN_RUN_RUNTIME_STACK_TESTS") != "1",
    reason="Requires Docker Compose PostgreSQL, MLflow and MinIO runtime stack.",
)


def test_runtime_stack_connectivity_and_artifact_write() -> None:
    stack = RuntimeStack(Settings(runtime_stack_enabled=True))

    status = stack.initialize()

    assert status.postgres.connected
    assert status.mlflow.connected
    assert status.minio.connected
    assert status.artifact_storage.connected


def test_runtime_stack_creates_mlflow_run() -> None:
    stack = RuntimeStack(Settings(runtime_stack_enabled=True))

    run_id = stack.start_mlflow_run(
        run_name="runtime-stack-integration-test",
        params={"suite": "runtime-stack"},
    )
    try:
        assert run_id
        stack.log_mlflow_metric("validation_loss", 0.25, step=1)
        stack.log_mlflow_text("rollback_events.txt", "no rollback")
    finally:
        stack.end_mlflow_run()
