from fastapi.testclient import TestClient

from tests.conftest import MockSentimentModelService


def test_health_check_healthy(client: TestClient) -> None:
    """Test health check returns healthy status when model is loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["api_active"] is True
    assert data["model_loaded"] is True
    assert data["model_name"] == "test-mock-model"
    assert "version" in data
    assert "environment" in data


def test_health_check_degraded(client: TestClient) -> None:
    """Test health check differentiates when API is active but model is not loaded."""
    # Temporarily degrade service
    client.app.state.sentiment_service = MockSentimentModelService(is_ready=False)

    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "degraded"
    assert data["api_active"] is True
    assert data["model_loaded"] is False


def test_health_check_v1_prefix(client: TestClient) -> None:
    """Verify health endpoint is also accessible under /api/v1 prefix."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
