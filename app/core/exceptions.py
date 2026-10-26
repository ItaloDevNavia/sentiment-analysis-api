"""Centralized exception handling for the FastAPI application."""
import logging
from typing import Any, Dict, List

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.services.sentiment_model import (
    InvalidScoreError,
    ModelLoadError,
    ModelNotReadyError,
    SentimentInferenceError,
    UnrecognizedModelLabelError,
)

logger = logging.getLogger(__name__)


def setup_exception_handlers(app: FastAPI) -> None:
    """Register centralized exception handlers on the FastAPI application."""

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Handle schema validation errors and return sanitized 422 responses."""
        sanitized_errors: List[Dict[str, Any]] = []
        for err in exc.errors():
            sanitized_errors.append(
                {
                    "loc": [str(loc_item) for loc_item in err.get("loc", []) if loc_item != "body"],
                    "msg": err.get("msg", "Validation error"),
                    "type": err.get("type", "value_error"),
                }
            )

        logger.warning(
            "Validation failed on %s %s: %d error(s)",
            request.method,
            request.url.path,
            len(sanitized_errors),
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": sanitized_errors},
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        """Handle standard HTTP exceptions while preserving status code and detail message."""
        logger.info(
            "HTTP exception on %s %s: status %d - %s",
            request.method,
            request.url.path,
            exc.status_code,
            exc.detail,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
        )

    @app.exception_handler(ModelNotReadyError)
    async def model_not_ready_handler(
        request: Request, exc: ModelNotReadyError
    ) -> JSONResponse:
        """Handle requests attempted before model is initialized or during degraded state."""
        logger.warning("Service rejected request on %s %s: model not ready", request.method, request.url.path)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "Model service is not ready. Please try again shortly."},
        )

    @app.exception_handler(ModelLoadError)
    async def model_load_error_handler(
        request: Request, exc: ModelLoadError
    ) -> JSONResponse:
        """Handle failures in loading the ML model."""
        logger.error("ModelLoadError on %s %s: %s", request.method, request.url.path, exc)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "Model loading failed. Service is currently unavailable."},
        )

    @app.exception_handler(UnrecognizedModelLabelError)
    @app.exception_handler(InvalidScoreError)
    @app.exception_handler(SentimentInferenceError)
    async def sentiment_inference_error_handler(
        request: Request, exc: SentimentInferenceError
    ) -> JSONResponse:
        """Handle internal machine learning inference errors without exposing model internals."""
        logger.error(
            "Inference error on %s %s: %s",
            request.method,
            request.url.path,
            exc.__class__.__name__,
            exc_info=True,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "An internal error occurred during sentiment classification."},
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Catch-all handler for unexpected server exceptions.

        Logs the full exception trace on the server for debugging without logging
        sensitive user payload text. Returns a clean, generic 500 message to the client.
        """
        logger.error(
            "Unhandled exception on %s %s: %s",
            request.method,
            request.url.path,
            exc,
            exc_info=True,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "An internal server error occurred."},
        )
