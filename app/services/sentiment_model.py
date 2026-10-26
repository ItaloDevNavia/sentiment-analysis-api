"""Sentiment analysis inference service wrapping Hugging Face Transformers pipeline."""
import logging
import math
from typing import Any, Callable, Dict, Optional, Tuple

from transformers import pipeline

from app.config import get_settings
from app.schemas.sentiment import SentimentLabel, SentimentResponse

logger = logging.getLogger(__name__)

# Explicit mapping of common transformer sentiment classification labels
_LABEL_MAPPING: Dict[str, SentimentLabel] = {
    # Textual forms
    "positive": SentimentLabel.POSITIVE,
    "pos": SentimentLabel.POSITIVE,
    "negative": SentimentLabel.NEGATIVE,
    "neg": SentimentLabel.NEGATIVE,
    "neutral": SentimentLabel.NEUTRAL,
    "neu": SentimentLabel.NEUTRAL,
    # Numerical index forms (cardiffnlp, bert-base-multilingual-uncased-sentiment, etc.)
    "label_0": SentimentLabel.NEGATIVE,
    "label_1": SentimentLabel.NEUTRAL,
    "label_2": SentimentLabel.POSITIVE,
}


# ============================================================================
# Exception Hierarchy for Sentiment Analysis Service
# ============================================================================

class SentimentServiceError(Exception):
    """Base exception for all sentiment service-related errors."""


class ModelLoadError(SentimentServiceError):
    """Raised when the ML model or pipeline fails to load during initialization."""


class SentimentInferenceError(SentimentServiceError):
    """Base exception for errors occurring during inference."""


class ModelNotReadyError(SentimentInferenceError):
    """Raised when inference is requested but the model has not been loaded yet."""


class UnrecognizedModelLabelError(SentimentInferenceError):
    """Raised when the underlying model returns a label that cannot be mapped to SentimentLabel."""


class InvalidScoreError(SentimentInferenceError):
    """Raised when the confidence score is NaN, infinite, or outside the [0.0, 1.0] interval."""


# ============================================================================
# Core Sentiment Model Service
# ============================================================================

class SentimentModelService:
    """Service encapsulating the lifecycle, loading, and inference of the sentiment model.

    Designed to be easily instantiated, managed via FastAPI lifespan, and swapped
    with mocks or alternate implementations during testing.
    """

    def __init__(self, pipeline_factory: Optional[Callable[..., Any]] = None) -> None:
        """Initialize the service.

        Args:
            pipeline_factory: Optional factory callable for creating the pipeline.
                              Defaults to transformers.pipeline. Useful for dependency injection.
        """
        self._pipeline: Optional[Any] = None
        self._model_name: Optional[str] = None
        self._pipeline_factory = pipeline_factory or pipeline

    @property
    def is_ready(self) -> bool:
        """Return True if the model pipeline is loaded and ready for inference."""
        return self._pipeline is not None

    @property
    def model_name(self) -> Optional[str]:
        """Return the identifier of the loaded model, or None if uninitialized."""
        return self._model_name

    def load_model(self, model_name: Optional[str] = None, force_reload: bool = False) -> None:
        """Load the sentiment analysis model pipeline into memory.

        Args:
            model_name: Identifier of the Hugging Face model or local path.
                        If None, reads from app configuration.
            force_reload: If True, reloads the model even if already initialized.

        Raises:
            ModelLoadError: If loading or instantiating the model pipeline fails.
        """
        target_model = model_name or get_settings().model_name

        if self.is_ready and not force_reload and self._model_name == target_model:
            logger.info("Model '%s' is already loaded. Skipping duplicate load.", target_model)
            return

        logger.info("Loading sentiment analysis model: '%s'...", target_model)
        try:
            self._pipeline = self._pipeline_factory(
                task="sentiment-analysis",
                model=target_model,
                tokenizer=target_model,
                truncation=True,
            )
            self._model_name = target_model
            logger.info("Sentiment analysis model '%s' loaded successfully.", target_model)
        except Exception as exc:
            self._pipeline = None
            self._model_name = None
            logger.error("Failed to load sentiment model '%s': %s", target_model, exc, exc_info=True)
            raise ModelLoadError(f"Could not load sentiment model '{target_model}': {exc}") from exc

    def _normalize_label(self, raw_label: str) -> SentimentLabel:
        """Map raw model label to the standardized SentimentLabel enum.

        Raises:
            UnrecognizedModelLabelError: If the label does not match known formats.
        """
        cleaned = raw_label.lower().strip()
        label = _LABEL_MAPPING.get(cleaned)
        if label is None:
            raise UnrecognizedModelLabelError(
                f"Model returned unrecognized sentiment label '{raw_label}'. "
                f"Supported labels: {[lbl.value for lbl in SentimentLabel]}"
            )
        return label

    def _validate_score(self, raw_score: float) -> float:
        """Validate that the score is a finite real number between 0.0 and 1.0.

        Raises:
            InvalidScoreError: If the score is NaN, infinite, or out of range.
        """
        try:
            score = float(raw_score)
        except (TypeError, ValueError) as exc:
            raise InvalidScoreError(f"Inference score is not a valid float: {raw_score}") from exc

        if not math.isfinite(score):
            raise InvalidScoreError(f"Inference score must be a finite number, got {score}")

        if not (0.0 <= score <= 1.0):
            raise InvalidScoreError(f"Inference score must be between 0.0 and 1.0, got {score}")

        return round(score, 4)

    def predict(self, text: str) -> SentimentResponse:
        """Perform sentiment inference on validated text.

        Args:
            text: Validated input string.

        Returns:
            SentimentResponse object containing text, sentiment label, and score.

        Raises:
            ModelNotReadyError: If the model has not been loaded.
            SentimentInferenceError: If model inference or parsing fails.
        """
        if not self.is_ready or self._pipeline is None:
            raise ModelNotReadyError(
                "Sentiment model is not loaded. Call 'load_model()' before running inference."
            )

        try:
            raw_output = self._pipeline(text)
            # transformers sentiment-analysis pipeline returns a list of dicts, e.g. [{'label': 'positive', 'score': 0.98}]
            if not raw_output or not isinstance(raw_output, list) or not isinstance(raw_output[0], dict):
                raise SentimentInferenceError(f"Unexpected pipeline output structure: {raw_output}")

            first_result = raw_output[0]
            raw_label = str(first_result.get("label", ""))
            raw_score = first_result.get("score")

            normalized_label = self._normalize_label(raw_label)
            validated_score = self._validate_score(raw_score)

            return SentimentResponse(
                text=text,
                sentiment=normalized_label,
                score=validated_score,
            )
        except SentimentServiceError:
            # Re-raise known domain service exceptions
            raise
        except Exception as exc:
            logger.error("Unexpected error during sentiment inference: %s", exc, exc_info=True)
            raise SentimentInferenceError(f"Inference failed unexpectedly: {exc}") from exc
