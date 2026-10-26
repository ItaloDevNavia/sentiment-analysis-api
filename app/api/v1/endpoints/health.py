"""Health check endpoint distinguishing between API liveness and model readiness."""
from fastapi import APIRouter, Depends, Request, Response, status

from app.config import Settings, get_settings
from app.schemas.health import HealthResponse
from app.services.sentiment_model import SentimentModelService

router = APIRouter()


def get_sentiment_service(request: Request) -> SentimentModelService:
    """Retrieve sentiment model service stored in FastAPI app.state."""
    return getattr(request.app.state, "sentiment_service", None)


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health and readiness check",
    description="Returns liveness of the API server and readiness of the ML sentiment model.",
    responses={
        200: {"description": "Service is healthy and model is ready for inference."},
        503: {"description": "API is active but model is uninitialized or degraded."},
    },
)
async def health_check(
    response: Response,
    settings: Settings = Depends(get_settings),
    service: SentimentModelService = Depends(get_sentiment_service),
) -> HealthResponse:
    """Evaluate service health and set appropriate HTTP status code."""
    model_loaded = service.is_ready if service is not None else False
    model_name = service.model_name if (service is not None and service.model_name) else settings.model_name

    if model_loaded:
        overall_status = "healthy"
        response.status_code = status.HTTP_200_OK
    else:
        overall_status = "degraded"
        # Semantics: return 503 Service Unavailable when the model is not ready
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return HealthResponse(
        status=overall_status,
        api_active=True,
        model_loaded=model_loaded,
        model_name=model_name,
        environment=settings.environment,
        version=settings.app_version,
    )
