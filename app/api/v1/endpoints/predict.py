"""Predict endpoint for sentiment classification."""
import logging
from fastapi import APIRouter, Depends, Request, status

from app.schemas.sentiment import SentimentRequest, SentimentResponse
from app.services.sentiment_model import ModelNotReadyError, SentimentModelService

logger = logging.getLogger(__name__)

router = APIRouter()


def get_ready_sentiment_service(request: Request) -> SentimentModelService:
    """Dependency that ensures SentimentModelService is present in app.state and ready."""
    service = getattr(request.app.state, "sentiment_service", None)
    if service is None or not service.is_ready:
        raise ModelNotReadyError(
            "Sentiment classification model is not ready. Please try again shortly."
        )
    return service


@router.post(
    "/predict",
    response_model=SentimentResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze text sentiment",
    description="Classifies input text as positive, negative, or neutral with a calibrated confidence score.",
    responses={
        200: {"description": "Successful sentiment analysis."},
        422: {"description": "Input validation error (empty string, whitespace only, or length exceeded)."},
        503: {"description": "Model service not ready or uninitialized."},
        500: {"description": "Internal error during inference."},
    },
)
async def predict_sentiment(
    payload: SentimentRequest,
    service: SentimentModelService = Depends(get_ready_sentiment_service),
) -> SentimentResponse:
    """Process incoming text and perform sentiment prediction."""
    return service.predict(payload.text)
