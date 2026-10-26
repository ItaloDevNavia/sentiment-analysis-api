"""Unit tests for SentimentModelService using mocks without network access."""
from unittest.mock import MagicMock
import pytest

from app.schemas.sentiment import SentimentLabel, SentimentResponse
from app.services.sentiment_model import (
    InvalidScoreError,
    ModelLoadError,
    ModelNotReadyError,
    SentimentInferenceError,
    SentimentModelService,
    UnrecognizedModelLabelError,
)


# ============================================================================
# Service Lifecycle & Readiness Tests
# ============================================================================

def test_service_initial_state_not_ready() -> None:
    """Verify that a newly instantiated service is not ready and model_name is None."""
    service = SentimentModelService()
    assert service.is_ready is False
    assert service.model_name is None


def test_predict_before_loading_raises_model_not_ready_error() -> None:
    """Verify calling predict on uninitialized service raises ModelNotReadyError."""
    service = SentimentModelService()
    with pytest.raises(ModelNotReadyError) as exc_info:
        service.predict("Any valid text")
    assert "Sentiment model is not loaded" in str(exc_info.value)


def test_service_load_model_success() -> None:
    """Verify successful loading sets is_ready to True and records model_name."""
    mock_pipeline = MagicMock()
    mock_factory = MagicMock(return_value=mock_pipeline)

    service = SentimentModelService(pipeline_factory=mock_factory)
    service.load_model("test-model")

    assert service.is_ready is True
    assert service.model_name == "test-model"
    mock_factory.assert_called_once_with(
        task="sentiment-analysis",
        model="test-model",
        tokenizer="test-model",
        truncation=True,
    )


def test_service_prevents_accidental_duplicate_load() -> None:
    """Verify that calling load_model repeatedly with same model name skips redundant reloading."""
    mock_factory = MagicMock(return_value=MagicMock())
    service = SentimentModelService(pipeline_factory=mock_factory)

    service.load_model("test-model")
    assert mock_factory.call_count == 1

    # Second load with same name should be skipped
    service.load_model("test-model", force_reload=False)
    assert mock_factory.call_count == 1

    # With force_reload=True, it should execute reload
    service.load_model("test-model", force_reload=True)
    assert mock_factory.call_count == 2


def test_service_load_model_failure_raises_model_load_error() -> None:
    """Verify that failure in pipeline creation raises ModelLoadError and leaves service unready."""
    def failing_factory(**kwargs):
        raise OSError("Model checkpoint not found or corrupted.")

    service = SentimentModelService(pipeline_factory=failing_factory)

    with pytest.raises(ModelLoadError) as exc_info:
        service.load_model("invalid-model-name")

    assert "Could not load sentiment model" in str(exc_info.value)
    assert service.is_ready is False
    assert service.model_name is None


# ============================================================================
# Inference & Label Normalization Tests
# ============================================================================

@pytest.mark.parametrize(
    "raw_label, expected_enum",
    [
        ("positive", SentimentLabel.POSITIVE),
        ("POSITIVE", SentimentLabel.POSITIVE),
        ("pos", SentimentLabel.POSITIVE),
        ("LABEL_2", SentimentLabel.POSITIVE),
        ("label_2", SentimentLabel.POSITIVE),
        ("negative", SentimentLabel.NEGATIVE),
        ("NEGATIVE", SentimentLabel.NEGATIVE),
        ("neg", SentimentLabel.NEGATIVE),
        ("LABEL_0", SentimentLabel.NEGATIVE),
        ("label_0", SentimentLabel.NEGATIVE),
        ("neutral", SentimentLabel.NEUTRAL),
        ("NEUTRAL", SentimentLabel.NEUTRAL),
        ("neu", SentimentLabel.NEUTRAL),
        ("LABEL_1", SentimentLabel.NEUTRAL),
        ("label_1", SentimentLabel.NEUTRAL),
    ],
)
def test_predict_normalizes_known_labels(raw_label: str, expected_enum: SentimentLabel) -> None:
    """Verify all recognized label variations map to the correct SentimentLabel enum."""
    mock_pipeline = MagicMock(return_value=[{"label": raw_label, "score": 0.9234}])
    service = SentimentModelService(pipeline_factory=lambda **kwargs: mock_pipeline)
    service.load_model("mock-model")

    result = service.predict("Sample text")

    assert isinstance(result, SentimentResponse)
    assert result.text == "Sample text"
    assert result.sentiment == expected_enum
    assert result.score == 0.9234


def test_predict_raises_error_on_unrecognized_label() -> None:
    """Verify that an unexpected label raises UnrecognizedModelLabelError instead of guessing."""
    mock_pipeline = MagicMock(return_value=[{"label": "IRRELEVANT_LABEL", "score": 0.88}])
    service = SentimentModelService(pipeline_factory=lambda **kwargs: mock_pipeline)
    service.load_model("mock-model")

    with pytest.raises(UnrecognizedModelLabelError) as exc_info:
        service.predict("Sample text")

    assert "unrecognized sentiment label 'IRRELEVANT_LABEL'" in str(exc_info.value)


# ============================================================================
# Score Validation Tests
# ============================================================================

@pytest.mark.parametrize("invalid_score", [1.05, -0.01, -100.0, float("nan"), float("inf")])
def test_predict_raises_error_on_invalid_or_non_finite_scores(invalid_score: float) -> None:
    """Verify that non-finite or out-of-range scores from pipeline raise InvalidScoreError."""
    mock_pipeline = MagicMock(return_value=[{"label": "positive", "score": invalid_score}])
    service = SentimentModelService(pipeline_factory=lambda **kwargs: mock_pipeline)
    service.load_model("mock-model")

    with pytest.raises(InvalidScoreError):
        service.predict("Sample text")


def test_predict_raises_inference_error_on_corrupt_pipeline_output() -> None:
    """Verify unexpected return structure from pipeline raises SentimentInferenceError."""
    mock_pipeline = MagicMock(return_value=[])  # Empty list
    service = SentimentModelService(pipeline_factory=lambda **kwargs: mock_pipeline)
    service.load_model("mock-model")

    with pytest.raises(SentimentInferenceError) as exc_info:
        service.predict("Sample text")

    assert "Unexpected pipeline output structure" in str(exc_info.value)
