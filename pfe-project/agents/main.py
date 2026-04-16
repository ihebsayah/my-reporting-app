"""Agent Service — entry point for the standalone FastAPI microservice.

Runs on port 8001 (separate from the existing app on port 8000).

Usage:
    uvicorn agents.main:app --host 0.0.0.0 --port 8001 --reload

Or via Docker Compose (see deploy/docker-compose.yml).

The agent service exposes these routes (all defined in agents/api/extraction_agent.py):
- POST /agents/extract          — main extraction + decision endpoint
- POST /agents/feedback         — record human feedback (real-time learning)
- GET  /agents/status           — health + enabled/disabled state
- GET  /agents/monitoring/accuracy      — sliding-window accuracy stats
- GET  /agents/monitoring/rollback-status
- POST /agents/admin/disable    — disable agents at runtime
- POST /agents/admin/enable     — re-enable agents at runtime
"""

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agents.api.extraction_agent import router as agent_router
from agents.config import get_agent_settings

_settings = get_agent_settings()

# ── Configure logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, _settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Application factory ────────────────────────────────────────────────────────
app = FastAPI(
    title="PFE — AI Agent Service",
    description=(
        "Intelligent extraction decision layer: 1 Master Agent + 4 Sub-Agents "
        "(Classifier, Extractor, Validator, Router). "
        "Augments the existing FastAPI extraction pipeline without replacing it."
    ),
    version=_settings.agent_service_version,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(agent_router)


# ── Global exception handler ───────────────────────────────────────────────────


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler — never expose raw tracebacks to callers."""
    logger.exception(
        "Unhandled exception in agent service for %s %s.",
        request.method,
        request.url.path,
        exc_info=exc,
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Agent service internal error. Document routed to human_review as safe default.",
            "error_code": "agent_service_error",
        },
    )


# ── Health check ───────────────────────────────────────────────────────────────


@app.get("/health", tags=["health"])
def health() -> dict:
    """Basic health check used by Docker Compose and load balancers.

    Returns:
        Status ``ok`` and service version.
    """
    from agents.monitoring.auto_rollback import AutoRollbackMonitor

    return {
        "status": "ok",
        "service": "agent-service",
        "version": _settings.agent_service_version,
        "agents_enabled": AutoRollbackMonitor.agents_enabled(),
    }


# ── Startup / shutdown events ──────────────────────────────────────────────────


@app.on_event("startup")
def on_startup() -> None:
    """Log startup and confirm agent and memory connectivity."""
    logger.info(
        "Agent Service starting on %s:%d (agents_enabled=%s, llm=%s).",
        _settings.agent_service_host,
        _settings.agent_service_port,
        _settings.agents_enabled,
        _settings.llm_model_name,
    )
    # Probe Redis (non-fatal).
    try:
        from agents.memory.redis_client import get_redis_client

        client = get_redis_client()
        if client:
            logger.info("Startup: Redis connection OK.")
        else:
            logger.warning("Startup: Redis unavailable — short-term memory will be in-process only.")
    except Exception as exc:
        logger.warning("Startup: Redis probe failed: %s.", exc)

    # Probe upstream FastAPI (non-fatal).
    try:
        import httpx

        resp = httpx.get(f"{_settings.api_base_url}/health", timeout=5)
        if resp.status_code == 200:
            logger.info("Startup: Upstream FastAPI at %s is reachable.", _settings.api_base_url)
        else:
            logger.warning(
                "Startup: Upstream FastAPI returned %d — ML tool calls may fail.",
                resp.status_code,
            )
    except Exception as exc:
        logger.warning(
            "Startup: Upstream FastAPI at %s unreachable: %s. ML tools will fail gracefully.",
            _settings.api_base_url, exc,
        )


@app.on_event("shutdown")
def on_shutdown() -> None:
    """Log clean shutdown."""
    logger.info("Agent Service shutting down.")
