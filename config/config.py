"""
Configuration Management - Pydantic BaseSettings Implementation
Laadt configuratie uit .env bestand zoals gespecificeerd in FASE 1 plan
"""

from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from typing import Optional


class Settings(BaseSettings):
    """
    Application settings using Pydantic BaseSettings
    Laadt alle variabelen uit .env bestand
    """

    # Database configuratie
    database_url: str

    # Redis configuratie
    redis_url: str

    # Security configuratie
    secret_key: str

    # Optionele configuratie
    environment: str = "development"
    debug: bool = False

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"  # Ignore extra environment variables
    )


# Global settings instance (lazy initialization)
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get application settings"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings