"""Unit tests for Pydantic schemas without HTTP endpoints."""
import pytest
from pydantic import ValidationError

from app.config import get_settings
from app.schemas.health import HealthResponse
from app.schemas.sentiment import SentimentLabel, SentimentRequest, SentimentResponse


# ============================================================================
# SentimentLabel Tests
# ============================================================================

def test_sentiment_label_allowed_values() -> None:
    """Verify that only positive, negative, and neutral exist as allowed labels."""
    allowed = {label.value for label in SentimentLabel}
    assert allowed == {"positive", "negative", "neutral"}


# ============================================================================
# SentimentRequest Tests
# ============================================================================

def test_sentiment_request_valid_text() -> None:
    """Verify valid text input is accepted and original text is preserved without silent mutation."""
    raw_text = "  FastAPI is very reliable.  "
    req = SentimentRequest(text=raw_text)
    # The text must not be stripped silently
    assert req.text == raw_text


def test_sentiment_request_rejects_empty_string() -> None:
    """Verify empty string raises a ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        SentimentRequest(text="")
    assert "Input text cannot be empty or contain only whitespace" in str(exc_info.value)


def test_sentiment_request_rejects_whitespace_only() -> None:
    """Verify whitespace-only string raises a ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        SentimentRequest(text="   \t\n   ")
    assert "Input text cannot be empty or contain only whitespace" in str(exc_info.value)


def test_sentiment_request_rejects_text_exceeding_max_length() -> None:
    """Verify text exceeding max_text_length from settings raises a ValidationError."""
    settings = get_settings()
    max_len = settings.max_text_length
    exceeding_text = "a" * (max_len + 1)

    with pytest.raises(ValidationError) as exc_info:
        SentimentRequest(text=exceeding_text)
    assert f"exceeds maximum allowed length of {max_len}" in str(exc_info.value)


def test_sentiment_request_accepts_text_at_exact_max_length() -> None:
    """Verify boundary condition where text length is exactly max_text_length."""
    settings = get_settings()
    boundary_text = "x" * settings.max_text_length
    req = SentimentRequest(text=boundary_text)
    assert len(req.text) == settings.max_text_length


# ============================================================================
# SentimentResponse Tests
# ============================================================================

@pytest.mark.parametrize("label", [SentimentLabel.POSITIVE, SentimentLabel.NEGATIVE, SentimentLabel.NEUTRAL])
def test_sentiment_response_valid_labels(label: SentimentLabel) -> None:
    """Verify each allowed sentiment label can instantiate a valid SentimentResponse."""
    resp = SentimentResponse(text="Test text", sentiment=label, score=0.85)
    assert resp.sentiment == label
    assert resp.score == 0.85


def test_sentiment_response_score_boundaries() -> None:
    """Verify boundary scores (0.0 and 1.0) are valid."""
    resp_zero = SentimentResponse(text="T", sentiment=SentimentLabel.NEUTRAL, score=0.0)
    resp_one = SentimentResponse(text="T", sentiment=SentimentLabel.POSITIVE, score=1.0)
    assert resp_zero.score == 0.0
    assert resp_one.score == 1.0


@pytest.mark.parametrize("invalid_score", [-0.01, 1.01, -10.0, 50.0])
def test_sentiment_response_rejects_out_of_range_score(invalid_score: float) -> None:
    """Verify scores outside [0.0, 1.0] raise ValidationError."""
    with pytest.raises(ValidationError):
        SentimentResponse(text="T", sentiment=SentimentLabel.POSITIVE, score=invalid_score)


@pytest.mark.parametrize("non_finite_score", [float("nan"), float("inf"), float("-inf")])
def test_sentiment_response_rejects_non_finite_scores(non_finite_score: float) -> None:
    """Verify non-finite scores (NaN, inf, -inf) raise ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        SentimentResponse(text="T", sentiment=SentimentLabel.NEUTRAL, score=non_finite_score)
    assert "Sentiment score must be a finite real number" in str(exc_info.value)


# ============================================================================
# HealthResponse Tests
# ============================================================================

def test_health_response_schema() -> None:
    """Verify HealthResponse instantiates with valid types."""
    resp = HealthResponse(
        status="healthy",
        api_active=True,
        model_loaded=True,
        model_name="cardiffnlp/twitter-roberta-base-sentiment-latest",
        environment="development",
        version="1.0.0",
    )
    assert resp.status == "healthy"
    assert resp.api_active is True
    assert resp.model_loaded is True


@pytest.mark.parametrize("invalid_text", [None, [], {}, 12345])
def test_sentiment_request_rejects_non_string_types(invalid_text: object) -> None:
    """Verify non-string types for text field raise ValidationError."""
    with pytest.raises(ValidationError):
        SentimentRequest(text=invalid_text)  # type: ignore[arg-type]


def test_sentiment_response_rejects_non_convertible_score_type() -> None:
    """Verify string score that cannot be converted to float raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        SentimentResponse(text="Valid", sentiment=SentimentLabel.POSITIVE, score="not_a_float")  # type: ignore[arg-type]
    assert "Sentiment score must be a valid float" in str(exc_info.value)

