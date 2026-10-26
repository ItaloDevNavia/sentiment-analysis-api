from typing import Generator
import pytest
from fastapi.testclient import TestClient

from app.main import create_application
from app.schemas.sentiment import SentimentLabel
from app.services.sentiment_model import SentimentModelService


class MockSentimentModelService(SentimentModelService):
    """Mock implementation of SentimentModelService for deterministic and fast testing."""

    def __init__(self, is_ready: bool = True) -> None:
        super().__init__()
        self._is_ready = is_ready
        self._model_name = "test-mock-model"

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    @property
    def model_name(self) -> str:
        return self._model_name

    def load_model(self, model_name: str) -> None:
        self._is_ready = True
        self._model_name = model_name

    def predict(self, text: str) -> tuple[SentimentLabel, float]:
        if not self._is_ready:
            from app.services.sentiment_model import ModelNotReadyError
            raise ModelNotReadyError("Model is not ready.")

        lower_text = text.lower()
        if any(w in lower_text for w in ["great", "excellent", "love", "awesome", "good"]):
            return SentimentLabel.POSITIVE, 0.9850
        elif any(w in lower_text for w in ["terrible", "bad", "hate", "awful", "worst"]):
            return SentimentLabel.NEGATIVE, 0.9420
        return SentimentLabel.NEUTRAL, 0.7510


@pytest.fixture
def mock_service() -> MockSentimentModelService:
    """Fixture providing a ready mock sentiment service."""
    return MockSentimentModelService(is_ready=True)


@pytest.fixture
def client(mock_service: MockSentimentModelService) -> Generator[TestClient, None, None]:
    """TestClient fixture with mock model service injected into app state."""
    app = create_application()
    app.state.sentiment_service = mock_service
    with TestClient(app=app) as test_client:
        yield test_client
