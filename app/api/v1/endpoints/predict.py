import logging
from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.config import Settings, get_settings
from app.schemas.sentiment import SentimentRequest, SentimentResponse
from app.services.sentiment_model import ModelNotReadyError, SentimentModelService

logger = logging.getLogger(__name__)

router = APIRouter()


def get_sentiment_service_dep(request: Request) -> SentimentModelService:
    """Retrieve sentiment model service stored in FastAPI app state."""
    service = getattr(request.app.state, "sentiment_service", None)
    if service is None or not service.is_ready:
        raise ModelNotReadyError("Sentiment model is not initialized or ready for inference.")
    return service


@router.post(
    "/predict",
    response_model=SentimentResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze sentiment",
    description="Analyze text sentiment and return the label (positive, negative, neutral) with confidence score.",
    responses={
        200: {"description": "Sentiment successfully analyzed."},
        400: {"description": "Invalid input."},
        422: {"description": "Validation error (empty text or text exceeds maximum length)."},
        503: {"description": "Model is not loaded or ready."},
    },
)
async def predict_sentiment(
    payload: SentimentRequest,
    settings: Settings = Depends(get_settings),
    service: SentimentModelService = Depends(get_sentiment_service_dep),
) -> SentimentResponse:
    """Validate text and perform sentiment prediction."""
    # Check configured max length limit
    if len(payload.text) > settings.max_text_length:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Text exceeds the maximum allowed length of {settings.max_text_length} characters.",
        )

    sentiment, confidence = service.predict(payload.text)

    return SentimentResponse(
        text=payload.text,
        sentiment=sentiment,
        confidence=confidence,
    )
