"""Pytest configuration and fixtures for integration tests using httpx.AsyncClient."""
from typing import AsyncGenerator
import httpx
import pytest
import pytest_asyncio

from app.main import create_application
from app.schemas.sentiment import SentimentLabel, SentimentResponse
from app.services.sentiment_model import SentimentModelService


class MockSentimentService(SentimentModelService):
    """Mock implementation of SentimentModelService for fast, offline testing."""

    def __init__(self, is_ready: bool = True) -> None:
        super().__init__()
        self._is_ready = is_ready
        self._model_name = "mock-sentiment-model"

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    @property
    def model_name(self) -> str:
        return self._model_name

    def load_model(self, model_name: str = None, force_reload: bool = False) -> None:
        self._is_ready = True
        self._model_name = model_name or "mock-sentiment-model"

    def predict(self, text: str) -> SentimentResponse:
        from app.services.sentiment_model import ModelNotReadyError
        if not self._is_ready:
            raise ModelNotReadyError("Model is not ready.")

        lower = text.lower()
        if any(w in lower for w in ["great", "love", "amazing", "good"]):
            label = SentimentLabel.POSITIVE
            score = 0.9850
        elif any(w in lower for w in ["terrible", "bad", "hate", "awful"]):
            label = SentimentLabel.NEGATIVE
            score = 0.9420
        else:
            label = SentimentLabel.NEUTRAL
            score = 0.7500

        return SentimentResponse(text=text, sentiment=label, score=score)


@pytest.fixture
def ready_mock_service() -> MockSentimentService:
    """Fixture providing a ready MockSentimentService."""
    return MockSentimentService(is_ready=True)


@pytest_asyncio.fixture
async def client(ready_mock_service: MockSentimentService) -> AsyncGenerator[httpx.AsyncClient, None]:
    """Asynchronous HTTP client fixture executing within FastAPI's lifespan context."""
    app = create_application()
    app.state.sentiment_service = ready_mock_service
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as test_client:
            yield test_client
