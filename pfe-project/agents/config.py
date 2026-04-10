"""Agent service configuration — all tunable parameters in one place.

All values can be overridden via environment variables.  The agent service
intentionally re-uses the existing FastAPI base URL so it can call the
extraction pipeline as a tool without importing app code directly.
"""

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentSettings(BaseSettings):
    """Central configuration for the AI agent microservice."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Service identity ────────────────────────────────────────────────────
    agent_service_host: str = Field(default="0.0.0.0", alias="AGENT_SERVICE_HOST")
    agent_service_port: int = Field(default=8001, alias="AGENT_SERVICE_PORT")
    agent_service_version: str = Field(default="0.1.0", alias="AGENT_SERVICE_VERSION")

    # ── Feature flag: globally enable/disable agents ────────────────────────
    agents_enabled: bool = Field(default=True, alias="AGENTS_ENABLED")

    # ── Existing FastAPI base URL (agents call the ML pipeline as a tool) ──
    api_base_url: str = Field(default="http://localhost:8000", alias="API_BASE_URL")

    # ── LLM settings (open-source, Ollama-served) ──────────────────────────
    llm_base_url: str = Field(default="http://localhost:11434", alias="LLM_BASE_URL")
    llm_model_name: str = Field(default="mistral", alias="LLM_MODEL_NAME")
    llm_temperature: float = Field(default=0.1, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=1024, alias="LLM_MAX_TOKENS")
    llm_request_timeout: int = Field(default=60, alias="LLM_REQUEST_TIMEOUT")

    # ── Redis (real-time / short-term memory) ──────────────────────────────
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    redis_ttl_seconds: int = Field(default=604800, alias="REDIS_TTL_SECONDS")  # 7 days
    redis_pattern_ttl_seconds: int = Field(
        default=2592000, alias="REDIS_PATTERN_TTL_SECONDS"  # 30 days
    )

    # ── PostgreSQL (long-term memory — same DB as existing app) ────────────
    database_url: str = Field(
        default="sqlite:///./reporting_app.db",
        alias="DATABASE_URL",
    )

    # ── Router / routing thresholds ────────────────────────────────────────
    auto_approve_threshold: float = Field(
        default=0.85, alias="AGENT_AUTO_APPROVE_THRESHOLD"
    )
    human_review_threshold: float = Field(
        default=0.65, alias="AGENT_HUMAN_REVIEW_THRESHOLD"
    )

    # ── Safety rails ───────────────────────────────────────────────────────
    safety_max_amount: float = Field(
        default=100_000.0, alias="SAFETY_MAX_AMOUNT"
    )

    # ── Auto-rollback triggers ─────────────────────────────────────────────
    rollback_accuracy_threshold: float = Field(
        default=0.85, alias="ROLLBACK_ACCURACY_THRESHOLD"
    )
    rollback_override_rate_threshold: float = Field(
        default=0.30, alias="ROLLBACK_OVERRIDE_RATE_THRESHOLD"
    )
    rollback_window_size: int = Field(default=100, alias="ROLLBACK_WINDOW_SIZE")

    # ── Logging ────────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_agent_reasoning: bool = Field(default=True, alias="LOG_AGENT_REASONING")


@lru_cache(maxsize=1)
def get_agent_settings() -> AgentSettings:
    """Return a cached agent settings instance."""
    return AgentSettings()
