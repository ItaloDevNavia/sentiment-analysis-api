"""FastAPI application initialization with lifespan management and CORS."""
from contextlib import asynccontextmanager
import logging
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.config import get_settings
from app.core.exceptions import setup_exception_handlers
from app.services.sentiment_model import SentimentModelService

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sentiment_api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown lifecycle events.

    Instantiates and loads the SentimentModelService once at startup.
    If model loading fails, logs the critical error and leaves the service
    in a degraded (not ready) state.
    """
    logger.info("Initializing %s (environment: %s)...", settings.app_name, settings.environment)

    # Allow service pre-injection (useful for tests or custom factories)
    service = getattr(app.state, "sentiment_service", None)
    if service is None:
        service = SentimentModelService()
        app.state.sentiment_service = service

    # Load model from Hugging Face if not already loaded and not in testing mode
    if not service.is_ready and settings.environment.lower() != "testing":
        try:
            service.load_model(settings.model_name)
            logger.info("Model '%s' successfully loaded and ready for inference.", settings.model_name)
        except Exception as exc:
            logger.error(
                "CRITICAL: Failed to load sentiment model '%s' during application startup: %s. "
                "The application will start in a degraded state (not ready for inference).",
                settings.model_name,
                exc,
                exc_info=True,
            )

    yield

    logger.info("Application shutdown initiated. Cleaning up resources...")


def create_application() -> FastAPI:
    """Application factory for FastAPI instance."""
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Production-grade REST API for sentiment analysis using Hugging Face Transformers.",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Explicit and safe CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # Centralized exception handlers
    setup_exception_handlers(app)

    # Mount API routes
    app.include_router(api_router)

    return app


app = create_application()
