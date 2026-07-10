from fastapi.testclient import TestClient

from doe_service.main import create_app


def test_health_reports_ok_and_doe_version() -> None:
    client = TestClient(create_app())
    response = client.get("/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["doe_version"]
