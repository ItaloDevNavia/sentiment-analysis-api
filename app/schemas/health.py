from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Schema for health check endpoint."""

    status: str = Field(..., description="Overall health status of the service ('healthy' or 'degraded')")
    api_active: bool = Field(True, description="Indicates if the FastAPI application is alive and accepting requests")
    model_loaded: bool = Field(..., description="Indicates if the ML model is loaded into memory and ready for inference")
    model_name: str = Field(..., description="Name or identifier of the ML model")
    environment: str = Field(..., description="Application execution environment")
    version: str = Field(..., description="API version")

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
