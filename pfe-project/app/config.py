"""Application configuration."""

import json
from functools import lru_cache
from typing import Dict, List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = Field(default="PFE Reporting App", alias="APP_NAME")
    app_version: str = Field(default="0.1.0", alias="APP_VERSION")
    app_env: str = Field(default="development", alias="APP_ENV")
    app_debug: bool = Field(default=True, alias="APP_DEBUG")
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    database_url: str = Field(
        default="sqlite:///./reporting_app.db",
        alias="DATABASE_URL",
    )
    label_studio_url: str = Field(
        default="http://localhost:8080",
        alias="LABEL_STUDIO_URL",
    )
    label_studio_api_key: str = Field(
        default="change-me",
        alias="LABEL_STUDIO_API_KEY",
    )
    label_studio_project_title: str = Field(
        default="Invoice Field Extraction",
        alias="LABEL_STUDIO_PROJECT_TITLE",
    )
    default_locale: str = Field(default="en", alias="DEFAULT_LOCALE")
    default_timezone: str = Field(default="Africa/Tunis", alias="DEFAULT_TIMEZONE")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    auto_approval_threshold: float = Field(
        default=0.9,
        alias="AUTO_APPROVAL_THRESHOLD",
    )
    review_min_threshold: float = Field(
        default=0.7,
        alias="REVIEW_MIN_THRESHOLD",
    )
    review_max_threshold: float = Field(
        default=0.89,
        alias="REVIEW_MAX_THRESHOLD",
    )
    ner_model_output_dir: str = Field(
        default="artifacts/models/ner",
        alias="NER_MODEL_OUTPUT_DIR",
    )
    ner_train_iterations: int = Field(
        default=20,
        alias="NER_TRAIN_ITERATIONS",
    )
    ner_model_path: str = Field(
        default="artifacts/models/ner",
        alias="NER_MODEL_PATH",
    )
    pipeline_version: str = Field(
        default="sequential-v1",
        alias="PIPELINE_VERSION",
    )
    extraction_version: str = Field(
        default="regex-spacy-ensemble-v1",
        alias="EXTRACTION_VERSION",
    )
    model_version: str = Field(
        default="untrained-regex-fallback",
        alias="MODEL_VERSION",
    )
    field_thresholds_json: str = Field(
        default=(
            '{"INVOICE_ID":{"auto":0.92,"review_min":0.72,"review_max":0.91},'
            '"INVOICE_DATE":{"auto":0.90,"review_min":0.70,"review_max":0.89},'
            '"TOTAL_AMOUNT":{"auto":0.95,"review_min":0.80,"review_max":0.94},'
            '"VENDOR_NAME":{"auto":0.88,"review_min":0.68,"review_max":0.87}}'
        ),
        alias="FIELD_THRESHOLDS_JSON",
    )

    def cors_origins(self) -> List[str]:
        """Return the default list of allowed CORS origins.

        Returns:
            A permissive list in development mode and an empty list otherwise.
        """
        return ["*"] if self.app_debug else []

    def field_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Return parsed field-specific threshold configuration.

        Returns:
            Mapping of field names to threshold dictionaries.

        Raises:
            ValueError: If the configured JSON cannot be parsed.
        """
        try:
            payload = json.loads(self.field_thresholds_json)
        except json.JSONDecodeError as exc:
            raise ValueError("FIELD_THRESHOLDS_JSON must be valid JSON.") from exc
        if not isinstance(payload, dict):
            raise ValueError("FIELD_THRESHOLDS_JSON must be a JSON object.")
        return {
            str(field_name): {
                "auto": float(values["auto"]),
                "review_min": float(values["review_min"]),
                "review_max": float(values["review_max"]),
            }
            for field_name, values in payload.items()
            if isinstance(values, dict)
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings instance.

    Returns:
        Application settings loaded from environment variables and defaults.
    """
    return Settings()
