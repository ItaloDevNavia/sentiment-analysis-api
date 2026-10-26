"""Real model integration tests using live Hugging Face checkpoint.

Marked with @pytest.mark.integration so standard CI/unit test runs can exclude it:
    pytest -v -m "not integration"
Or run explicitly:
    pytest -v -m "integration"
"""
import pytest
from app.schemas.sentiment import SentimentLabel
from app.services.sentiment_model import SentimentModelService


@pytest.mark.integration
def test_real_model_inference_positive_sentiment() -> None:
    """Validate real Hugging Face model inference on clear positive text."""
    service = SentimentModelService()
    service.load_model("cardiffnlp/twitter-roberta-base-sentiment-latest")

    result = service.predict("FastAPI and machine learning are truly amazing and fantastic!")
    assert result.sentiment == SentimentLabel.POSITIVE
    assert 0.8 <= result.score <= 1.0


@pytest.mark.integration
def test_real_model_inference_negative_sentiment() -> None:
    """Validate real Hugging Face model inference on clear negative text."""
    service = SentimentModelService()
    service.load_model("cardiffnlp/twitter-roberta-base-sentiment-latest")

    result = service.predict("This was a terrible experience and the service was awful.")
    assert result.sentiment == SentimentLabel.NEGATIVE
    assert 0.8 <= result.score <= 1.0


@pytest.mark.integration
def test_real_model_inference_neutral_sentiment() -> None:
    """Validate real Hugging Face model inference on objective factual text."""
    service = SentimentModelService()
    service.load_model("cardiffnlp/twitter-roberta-base-sentiment-latest")

    result = service.predict("The flight departs at 10:30 AM tomorrow morning.")
    assert result.sentiment == SentimentLabel.NEUTRAL
    assert 0.7 <= result.score <= 1.0
