from fastapi import APIRouter, Depends, Request

from app.config import Settings, get_settings
from app.schemas.health import HealthResponse
from app.services.sentiment_model import SentimentModelService

router = APIRouter()


def get_sentiment_service_dep(request: Request) -> SentimentModelService:
    """Retrieve sentiment model service stored in FastAPI app state."""
    return getattr(request.app.state, "sentiment_service", None)


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check and readiness status",
    description="Returns API liveness and ML model readiness status.",
)
async def health_check(
    settings: Settings = Depends(get_settings),
    service: SentimentModelService = Depends(get_sentiment_service_dep),
) -> HealthResponse:
    """Check health and distinguish between API liveness and model readiness."""
    model_loaded = service.is_ready if service is not None else False
    model_name = service.model_name if service is not None else settings.model_name
    overall_status = "healthy" if model_loaded else "degraded"

    return HealthResponse(
        status=overall_status,
        api_active=True,
        model_loaded=model_loaded,
        model_name=model_name,
        environment=settings.environment,
        version=settings.app_version,
    )
