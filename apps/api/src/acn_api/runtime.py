from typing import Any

from fastapi import APIRouter, Request

from acn.runtime import RuntimeStack, RuntimeStatus

router = APIRouter(prefix="/api/v1/runtime", tags=["runtime"])


@router.get("/health")
async def runtime_health(request: Request) -> dict[str, Any]:
    settings = request.app.state.settings
    status = RuntimeStack(settings).initialize()
    return _status_payload(status)


def _status_payload(status: RuntimeStatus) -> dict[str, Any]:
    return {
        "postgres": {
            "connected": status.postgres.connected,
            "message": status.postgres.message,
        },
        "mlflow": {
            "connected": status.mlflow.connected,
            "message": status.mlflow.message,
        },
        "minio": {
            "connected": status.minio.connected,
            "message": status.minio.message,
        },
        "artifactStorage": {
            "connected": status.artifact_storage.connected,
            "message": status.artifact_storage.message,
        },
    }
