from fastapi.testclient import TestClient

from acn.config.settings import Settings
from acn_api.main import create_app


def test_health_returns_environment() -> None:
    app = create_app(Settings(env="test"))
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "environment": "test"}
