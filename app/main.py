import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.config import get_settings
from app.core.exceptions import setup_exception_handlers
from app.services.sentiment_model import get_sentiment_service

# Setup basic logging
settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sentiment_api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown lifecycle events.

    Loads the ML model once into memory at startup to avoid re-loading on each request.
    If a service is already attached to app.state (e.g. injected during tests), it will be reused.
    """
    logger.info("Initializing %s (env: %s)...", settings.app_name, settings.environment)

    # Check if service is already provided (e.g. test fixture or mock)
    existing_service = getattr(app.state, "sentiment_service", None)
    if existing_service is not None:
        sentiment_service = existing_service
    else:
        sentiment_service = get_sentiment_service()
        if settings.environment.lower() != "testing" and not sentiment_service.is_ready:
            try:
                sentiment_service.load_model(settings.model_name)
                logger.info("Model loaded and ready for inference.")
            except Exception as exc:
                logger.error(
                    "Model failed to load during startup: %s. API will start in degraded mode.",
                    exc,
                )

    # Attach service instance to application state for dependency injection
    app.state.sentiment_service = sentiment_service

    yield

    logger.info("Application shutting down. Freeing resources...")


def create_application() -> FastAPI:
    """Application factory for FastAPI instance."""
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Production-ready REST API for text sentiment analysis using Hugging Face Transformers.",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Cross-Origin Resource Sharing (CORS) setup
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register secure and sanitized exception handlers
    setup_exception_handlers(app)

    # Mount endpoints directly (e.g., /health, /predict) and with /api/v1 prefix
    app.include_router(api_router)
    app.include_router(api_router, prefix="/api/v1")

    return app


app = create_application()
