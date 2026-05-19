from fastapi.testclient import TestClient

from acn.config.settings import Settings
from acn_api.main import create_app


def test_dashboard_snapshot_contract_returns_frontend_shape() -> None:
    client = TestClient(create_app(Settings(env="test")))

    response = client.get("/api/v1/dashboard/snapshot")

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {
        "commitGraph",
        "branchGraph",
        "metricsTimeline",
        "experiments",
        "controllerDecisions",
        "rollbackHistory",
        "liveLogs",
    }
    assert payload["commitGraph"] == {"nodes": [], "edges": []}
    assert payload["branchGraph"]["nodes"][0]["id"] == "visual-demo"
    assert payload["experiments"][0]["id"] == "exp_visual_demo"


def test_dashboard_snapshot_allows_local_vite_origins() -> None:
    client = TestClient(create_app(Settings(env="test")))

    response = client.get(
        "/api/v1/dashboard/snapshot",
        headers={"Origin": "http://localhost:5176"},
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5176"


def test_override_submission_returns_accepted_response() -> None:
    client = TestClient(create_app(Settings(env="test")))

    response = client.post(
        "/api/v1/overrides",
        json={
            "decisionId": "dec_1",
            "approvedBy": "operator",
            "reason": "Demo override.",
            "ticketId": "ACN-1",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "accepted",
        "decisionId": "dec_1",
        "approvedBy": "operator",
        "ticketId": "ACN-1",
    }


def test_dashboard_websocket_sends_snapshot_event() -> None:
    client = TestClient(create_app(Settings(env="test")))

    with client.websocket_connect("/api/v1/dashboard/ws") as websocket:
        message = websocket.receive_json()

    assert message["type"] == "snapshot"
    assert message["payload"]["commitGraph"] == {"nodes": [], "edges": []}
    assert message["payload"]["branchGraph"]["nodes"][0]["id"] == "visual-demo"
