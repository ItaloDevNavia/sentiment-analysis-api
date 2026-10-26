from fastapi.testclient import TestClient

from app.config import get_settings
from tests.conftest import MockSentimentModelService


def test_predict_positive_sentiment(client: TestClient) -> None:
    """Test successful positive prediction."""
    payload = {"text": "I had an awesome and great experience with this service!"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == payload["text"]
    assert data["sentiment"] == "positive"
    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_negative_sentiment(client: TestClient) -> None:
    """Test successful negative prediction."""
    payload = {"text": "This was the worst and most terrible product ever."}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == payload["text"]
    assert data["sentiment"] == "negative"
    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_neutral_sentiment(client: TestClient) -> None:
    """Test successful neutral prediction."""
    payload = {"text": "The delivery arrived on Tuesday at 3pm."}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == payload["text"]
    assert data["sentiment"] == "neutral"
    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_empty_text_returns_422(client: TestClient) -> None:
    """Test empty string returns 422 Unprocessable Entity."""
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    # Ensure no internal filesystem paths leaked
    assert "C:\\" not in str(data)
    assert "/Users/" not in str(data)


def test_predict_whitespace_only_returns_422(client: TestClient) -> None:
    """Test whitespace-only string returns 422 Unprocessable Entity."""
    response = client.post("/predict", json={"text": "     \n\t   "})
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_predict_missing_text_field_returns_422(client: TestClient) -> None:
    """Test missing 'text' key returns 422."""
    response = client.post("/predict", json={"wrong_field": "hello"})
    assert response.status_code == 422


def test_predict_text_exceeding_max_length_returns_422(client: TestClient) -> None:
    """Test text longer than configured max length returns 422."""
    settings = get_settings()
    too_long_text = "word " * (settings.max_text_length + 10)
    response = client.post("/predict", json={"text": too_long_text})
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_predict_when_model_not_ready_returns_503(client: TestClient) -> None:
    """Test service returns 503 if model is not loaded."""
    client.app.state.sentiment_service = MockSentimentModelService(is_ready=False)
    response = client.post("/predict", json={"text": "Everything is great"})
    assert response.status_code == 503
    data = response.json()
    assert "detail" in data
    assert "Model service is currently unavailable" in data["detail"]


def test_predict_v1_alias(client: TestClient) -> None:
    """Verify endpoint is accessible under /api/v1/predict."""
    payload = {"text": "FastAPI with Hugging Face is awesome!"}
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["sentiment"] == "positive"
