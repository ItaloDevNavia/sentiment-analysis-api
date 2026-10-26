"""Health schema definitions."""
from typing import Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Schema representing API health and ML model readiness."""

    status: str = Field(
        ...,
        description="General operational status (e.g. 'healthy' or 'degraded')",
        examples=["healthy"],
    )
    api_active: bool = Field(
        True,
        description="Indicates whether the FastAPI HTTP server is active and responding",
    )
    model_loaded: bool = Field(
        False,
        description="Indicates whether the sentiment classification model is loaded and ready for inference",
    )
    model_name: Optional[str] = Field(
        None,
        description="Identifier of the configured sentiment analysis model",
        examples=["cardiffnlp/twitter-roberta-base-sentiment-latest"],
    )
    environment: str = Field(
        ...,
        description="Execution environment (development, staging, production, testing)",
        examples=["development"],
    )
    version: str = Field(
        ...,
        description="Application version string",
        examples=["1.0.0"],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "healthy",
                "api_active": True,
                "model_loaded": True,
                "model_name": "cardiffnlp/twitter-roberta-base-sentiment-latest",
                "environment": "development",
                "version": "1.0.0",
            }
        }
    }
