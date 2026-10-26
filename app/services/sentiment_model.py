import logging
from functools import lru_cache
from typing import Any, Optional

from transformers import pipeline

from app.schemas.sentiment import SentimentLabel

logger = logging.getLogger(__name__)

# Standard mapping for common Hugging Face sentiment models
_LABEL_MAPPING = {
    "positive": SentimentLabel.POSITIVE,
    "pos": SentimentLabel.POSITIVE,
    "label_2": SentimentLabel.POSITIVE,
    "neutral": SentimentLabel.NEUTRAL,
    "neu": SentimentLabel.NEUTRAL,
    "label_1": SentimentLabel.NEUTRAL,
    "negative": SentimentLabel.NEGATIVE,
    "neg": SentimentLabel.NEGATIVE,
    "label_0": SentimentLabel.NEGATIVE,
}


class ModelNotReadyError(RuntimeError):
    """Raised when inference is attempted on an uninitialized model."""
    pass


class SentimentModelService:
    """Service responsible for loading and running sentiment analysis model inference."""

    def __init__(self) -> None:
        self._pipeline: Optional[Any] = None
        self._model_name: Optional[str] = None

    @property
    def is_ready(self) -> bool:
        """Check if model pipeline is initialized and loaded in memory."""
        return self._pipeline is not None

    @property
    def model_name(self) -> str:
        """Return the name of the currently configured model."""
        return self._model_name or "uninitialized"

    def load_model(self, model_name: str) -> None:
        """Load Hugging Face pipeline once during application startup.

        Args:
            model_name: Hugging Face model identifier or local path.
        """
        logger.info("Loading sentiment analysis model: %s", model_name)
        try:
            self._pipeline = pipeline(
                task="sentiment-analysis",
                model=model_name,
                tokenizer=model_name,
                truncation=True,
            )
            self._model_name = model_name
            logger.info("Model %s loaded successfully.", model_name)
        except Exception as exc:
            self._pipeline = None
            self._model_name = None
            logger.error("Failed to load sentiment model: %s", exc, exc_info=True)
            raise

    def predict(self, text: str) -> tuple[SentimentLabel, float]:
        """Perform sentiment inference on input text.

        Args:
            text: Validated string to evaluate.

        Returns:
            Tuple of (SentimentLabel, confidence_score between 0.0 and 1.0).

        Raises:
            ModelNotReadyError: If model pipeline has not been loaded.
        """
        if not self.is_ready or self._pipeline is None:
            raise ModelNotReadyError("Sentiment model is not loaded or ready for inference.")

        # Hugging Face pipeline returns list of dicts, e.g. [{'label': 'positive', 'score': 0.98}]
        raw_output = self._pipeline(text)
        result = raw_output[0]
        raw_label = str(result.get("label", "")).lower().strip()
        confidence = float(result.get("score", 0.0))

        # Clamp confidence between 0.0 and 1.0
        clamped_confidence = max(0.0, min(1.0, round(confidence, 4)))

        # Normalize label to SentimentLabel enum
        normalized_label = _LABEL_MAPPING.get(raw_label)
        if normalized_label is None:
            logger.warning("Unrecognized label '%s' from model, defaulting to neutral", raw_label)
            normalized_label = SentimentLabel.NEUTRAL

        return normalized_label, clamped_confidence


# Singleton instance across application
_sentiment_service_instance = SentimentModelService()


def get_sentiment_service() -> SentimentModelService:
    """Return the singleton instance of SentimentModelService."""
    return _sentiment_service_instance
