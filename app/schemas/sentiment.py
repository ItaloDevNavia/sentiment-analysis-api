from enum import Enum
from pydantic import BaseModel, Field, field_validator


class SentimentLabel(str, Enum):
    """Sentiment classification labels supported by the API."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class SentimentRequest(BaseModel):
    """Payload schema for sentiment prediction request."""

    text: str = Field(
        ...,
        description="Text to analyze for sentiment",
        min_length=1,
        max_length=1000,
        examples=["This product exceeded my expectations and customer support was outstanding!"],
    )

    @field_validator("text")
    @classmethod
    def validate_not_blank(cls, value: str) -> str:
        """Ensure text is not just whitespace."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("Text cannot be empty or contain only whitespace.")
        return stripped


class SentimentResponse(BaseModel):
    """Prediction result response schema."""

    text: str = Field(..., description="The original or cleaned text analyzed")
    sentiment: SentimentLabel = Field(..., description="Predicted sentiment label")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for the predicted sentiment between 0.0 and 1.0",
        examples=[0.96],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "This product exceeded my expectations and customer support was outstanding!",
                "sentiment": "positive",
                "confidence": 0.9625,
            }
        }
    }
