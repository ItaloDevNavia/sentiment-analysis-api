"""Integration tests for HTTP endpoints using httpx.AsyncClient."""
from unittest.mock import MagicMock
import httpx
import pytest

from app.config import get_settings
from app.main import create_application
from app.services.sentiment_model import (
    ModelLoadError,
    SentimentInferenceError,
    SentimentModelService,
)
from tests.conftest import MockSentimentService


# ============================================================================
# GET /health Tests
# ============================================================================

@pytest.mark.asyncio
async def test_health_check_healthy(client: httpx.AsyncClient) -> None:
    """Verify /health returns 200 OK and healthy status when model is ready."""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["api_active"] is True
    assert data["model_loaded"] is True
    assert data["model_name"] == "mock-sentiment-model"
    assert "environment" in data
    assert "version" in data


@pytest.mark.asyncio
async def test_health_check_degraded_when_model_not_ready(client: httpx.AsyncClient) -> None:
    """Verify /health returns 503 Service Unavailable when model is not loaded."""
    client._transport.app.state.sentiment_service = MockSentimentService(is_ready=False)
    response = await client.get("/health")
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "degraded"
    assert data["api_active"] is True
    assert data["model_loaded"] is False


@pytest.mark.asyncio
async def test_health_check_v1_alias(client: httpx.AsyncClient) -> None:
    """Verify /api/v1/health is accessible and returns consistent response."""
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


# ============================================================================
# POST /predict Tests
# ============================================================================

@pytest.mark.asyncio
async def test_predict_valid_request(client: httpx.AsyncClient) -> None:
    """Verify POST /predict returns 200 and valid SentimentResponse."""
    payload = {"text": "This product is absolutely amazing and I love it!"}
    response = await client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == payload["text"]
    assert data["sentiment"] == "positive"
    assert 0.0 <= data["score"] <= 1.0


@pytest.mark.asyncio
async def test_predict_v1_alias(client: httpx.AsyncClient) -> None:
    """Verify /api/v1/predict endpoint alias works identically."""
    payload = {"text": "FastAPI is great!"}
    response = await client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["sentiment"] == "positive"


@pytest.mark.asyncio
async def test_predict_rejects_empty_text_returns_422(client: httpx.AsyncClient) -> None:
    """Verify empty text string returns 422 Unprocessable Entity."""
    response = await client.post("/predict", json={"text": ""})
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    assert "C:\\" not in str(data)
    assert "/Users/" not in str(data)


@pytest.mark.asyncio
async def test_predict_rejects_whitespace_only_returns_422(client: httpx.AsyncClient) -> None:
    """Verify whitespace-only text returns 422."""
    response = await client.post("/predict", json={"text": "     \n\t   "})
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


@pytest.mark.asyncio
async def test_predict_rejects_missing_text_field_returns_422(client: httpx.AsyncClient) -> None:
    """Verify missing text field in payload returns 422."""
    response = await client.post("/predict", json={"invalid_field": "some text"})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_predict_rejects_text_exceeding_max_length_returns_422(client: httpx.AsyncClient) -> None:
    """Verify text exceeding max_text_length returns 422."""
    settings = get_settings()
    too_long_text = "a" * (settings.max_text_length + 1)
    response = await client.post("/predict", json={"text": too_long_text})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_predict_when_model_not_ready_returns_503(client: httpx.AsyncClient) -> None:
    """Verify POST /predict returns 503 when model is not initialized."""
    client._transport.app.state.sentiment_service = MockSentimentService(is_ready=False)
    response = await client.post("/predict", json={"text": "Hello world"})
    assert response.status_code == 503
    data = response.json()
    assert "detail" in data
    assert "Model service is not ready" in data["detail"]


@pytest.mark.asyncio
async def test_predict_when_inference_fails_returns_500_without_leak() -> None:
    """Verify controlled 500 response on SentimentInferenceError without leaking internals."""
    failing_service = MagicMock(spec=SentimentModelService)
    failing_service.is_ready = True
    failing_service.predict.side_effect = SentimentInferenceError("Internal model tensor fault at /sys/path")

    app = create_application()
    app.state.sentiment_service = failing_service

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as custom_client:
            response = await custom_client.post("/predict", json={"text": "Analyze this text"})
            assert response.status_code == 500
            data = response.json()
            assert data["detail"] == "An internal error occurred during sentiment classification."
            assert "/sys/path" not in str(data)
            assert "tensor" not in str(data)


@pytest.mark.asyncio
async def test_unexpected_exception_returns_clean_500() -> None:
    """Verify unhandled generic Exception returns sanitized 500 without leaking stack traces."""
    faulty_service = MagicMock(spec=SentimentModelService)
    faulty_service.is_ready = True
    faulty_service.predict.side_effect = RuntimeError("Database connection string postgres://user:secret@db leaked!")

    app = create_application()
    app.state.sentiment_service = faulty_service

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as custom_client:
            response = await custom_client.post("/predict", json={"text": "Any text"})
            assert response.status_code == 500
            data = response.json()
            assert data["detail"] == "An internal server error occurred."
            assert "postgres" not in str(data)
            assert "secret" not in str(data)


# ============================================================================
# Lifespan Startup Failure Test
# ============================================================================

@pytest.mark.asyncio
async def test_lifespan_startup_failure_leaves_service_degraded() -> None:
    """Verify that if model loading fails during lifespan, app starts in degraded state."""
    failing_service = MagicMock(spec=SentimentModelService)
    failing_service.is_ready = False
    failing_service.model_name = None
    failing_service.load_model.side_effect = ModelLoadError("Network timeout downloading weights")

    app = create_application()
    app.state.sentiment_service = failing_service

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as custom_client:
            # App starts but /health reflects degraded state
            health_resp = await custom_client.get("/health")
            assert health_resp.status_code == 503
            assert health_resp.json()["status"] == "degraded"
            assert health_resp.json()["model_loaded"] is False

            # Prediction returns 503
            predict_resp = await custom_client.post("/predict", json={"text": "Test text"})
            assert predict_resp.status_code == 503


# ============================================================================
# Boundary & Additional HTTP Exception Tests
# ============================================================================

@pytest.mark.asyncio
async def test_predict_exact_max_length_accepted(client: httpx.AsyncClient) -> None:
    """Verify text of exactly max_text_length characters is accepted with 200 OK."""
    settings = get_settings()
    exact_text = "x" * settings.max_text_length
    response = await client.post("/predict", json={"text": exact_text})
    assert response.status_code == 200
    assert len(response.json()["text"]) == settings.max_text_length


@pytest.mark.parametrize("invalid_payload", [{"text": None}, {"text": []}, {"text": {}}])
@pytest.mark.asyncio
async def test_predict_rejects_non_string_payloads(client: httpx.AsyncClient, invalid_payload: dict) -> None:
    """Verify non-string types in JSON payload return 422."""
    response = await client.post("/predict", json=invalid_payload)
    assert response.status_code == 422
    assert "detail" in response.json()


@pytest.mark.asyncio
async def test_predict_rejects_malformed_json_body(client: httpx.AsyncClient) -> None:
    """Verify malformed JSON payload returns 422 with sanitized error format."""
    response = await client.post(
        "/predict",
        content=b"{invalid_json:",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
    assert "detail" in response.json()


@pytest.mark.asyncio
async def test_not_found_endpoint_returns_404_json(client: httpx.AsyncClient) -> None:
    """Verify non-existent route returns 404 handled by StarletteHTTPException handler."""
    response = await client.get("/non_existent_route")
    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert data["detail"] == "Not Found"


@pytest.mark.asyncio
async def test_method_not_allowed_returns_405_json(client: httpx.AsyncClient) -> None:
    """Verify invalid HTTP method returns 405 handled by StarletteHTTPException handler."""
    response = await client.delete("/predict")
    assert response.status_code == 405
    data = response.json()
    assert "detail" in data

