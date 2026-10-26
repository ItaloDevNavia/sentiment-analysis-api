"""Sentiment request and response schema definitions."""
from enum import Enum
import math
from pydantic import BaseModel, Field, field_validator

from app.config import get_settings


class SentimentLabel(str, Enum):
    """Supported sentiment classification labels."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class SentimentRequest(BaseModel):
    """Schema for incoming sentiment analysis inference requests."""

    text: str = Field(
        ...,
        description="Raw input text to be evaluated for sentiment.",
        examples=["FastAPI and modern machine learning make building APIs a joy!"],
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        """Validate that input text is non-empty, not whitespace-only, and within configured length limits.

        Note:
            The original text string is preserved exactly as submitted (no silent stripping or mutating)
            to ensure transparency and downstream reproducibility.
        """
        if not value or not value.strip():
            raise ValueError("Input text cannot be empty or contain only whitespace.")

        max_allowed = get_settings().max_text_length
        if len(value) > max_allowed:
            raise ValueError(
                f"Input text length ({len(value)}) exceeds maximum allowed length of {max_allowed} characters."
            )

        return value


class SentimentResponse(BaseModel):
    """Schema for sentiment prediction results."""

    text: str = Field(
        ...,
        description="The text analyzed.",
        examples=["FastAPI and modern machine learning make building APIs a joy!"],
    )
    sentiment: SentimentLabel = Field(
        ...,
        description="Predicted sentiment classification category.",
        examples=[SentimentLabel.POSITIVE],
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for the predicted sentiment between 0.0 and 1.0.",
        examples=[0.985],
    )

    @field_validator("score", mode="before")
    @classmethod
    def validate_finite_score(cls, value: float) -> float:
        """Ensure the score is a finite real number between 0.0 and 1.0."""
        try:
            num = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("Sentiment score must be a valid float.") from exc

        if not math.isfinite(num):
            raise ValueError("Sentiment score must be a finite real number.")
        if not (0.0 <= num <= 1.0):
            raise ValueError(f"Sentiment score must be between 0.0 and 1.0, got {num}.")
        return num

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "FastAPI and modern machine learning make building APIs a joy!",
                "sentiment": "positive",
                "score": 0.985,
            }
        }
    }
