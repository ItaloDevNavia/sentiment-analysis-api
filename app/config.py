from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables or .env file."""

    app_name: str = "Sentiment Analysis API"
    app_version: str = "1.0.0"
    environment: str = "development"
    log_level: str = "INFO"

    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8000

    # Model inference settings
    # CardiffNLP's roberta model outputs labels: positive, neutral, negative
    model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    max_text_length: int = 500

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached instance of the application settings."""
    return Settings()
