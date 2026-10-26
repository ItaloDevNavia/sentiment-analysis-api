import logging
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.services.sentiment_model import ModelNotReadyError

logger = logging.getLogger(__name__)


def setup_exception_handlers(app: FastAPI) -> None:
    """Register centralized exception handlers on the FastAPI application."""

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Handle Pydantic/FastAPI input validation errors with sanitized output."""
        sanitized_errors = []
        for error in exc.errors():
            # Keep field path and clear human-readable message without exposing internal code
            sanitized_errors.append(
                {
                    "loc": [str(loc_item) for loc_item in error.get("loc", []) if loc_item != "body"],
                    "msg": error.get("msg", "Invalid input value"),
                    "type": error.get("type", "value_error"),
                }
            )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": sanitized_errors},
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        """Handle standard HTTP exceptions preserving their respective status codes."""
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
        )

    @app.exception_handler(ModelNotReadyError)
    async def model_not_ready_handler(
        request: Request, exc: ModelNotReadyError
    ) -> JSONResponse:
        """Handle cases where the ML model is uninitialized or degraded."""
        logger.warning("Request rejected because model is not ready: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "detail": "Model service is currently unavailable. Please try again shortly."
            },
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(
        request: Request, exc: ValueError
    ) -> JSONResponse:
        """Handle business-level validation errors."""
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": str(exc)},
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Catch-all handler for unhandled exceptions.

        Logs full trace internally for debugging while returning safe generic error to client.
        Prevents leaking paths, database strings, or stack traces.
        """
        logger.error(
            "Unhandled server error on %s %s: %s",
            request.method,
            request.url.path,
            exc,
            exc_info=True,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "An internal server error occurred. Please contact the administrator if the problem persists."
            },
        )
