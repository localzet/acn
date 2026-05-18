from fastapi.testclient import TestClient

from acn.config.settings import Settings
from acn_api.main import create_app


def test_visual_demo_state_contract() -> None:
    client = TestClient(create_app(Settings(env="test")))

    response = client.get("/api/v1/demo/state")

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {
        "status",
        "autoMode",
        "epoch",
        "stage",
        "controllerState",
        "currentBranch",
        "activeCheckpointId",
        "rollbackCount",
        "gpuUsage",
        "metrics",
        "checkpoints",
        "predictions",
        "events",
        "decisions",
        "runtimeStatus",
        "mlflowRunId",
        "artifacts",
    }
    assert payload["currentBranch"] == "main"
    assert isinstance(payload["metrics"], list)
