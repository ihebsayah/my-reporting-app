"""FastAPI application entry point."""

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.routes import router as api_router
from app.api.schemas import ErrorResponse, HealthResponse
from app.config import get_settings
from app.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)
settings = get_settings()

app = FastAPI(title=settings.app_name, debug=settings.app_debug)
app.include_router(api_router, prefix="/api")


@app.exception_handler(HTTPException)
def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Return a standard JSON payload for handled HTTP exceptions.

    Args:
        request: Incoming request instance.
        exc: Raised HTTP exception.

    Returns:
        JSON response using the shared error schema.
    """
    logger.warning(
        "HTTP exception raised for %s %s: %s",
        request.method,
        request.url.path,
        exc.detail,
    )
    payload = ErrorResponse(
        detail=str(exc.detail),
        error_code=f"http_{exc.status_code}",
    )
    return JSONResponse(status_code=exc.status_code, content=payload.model_dump())


@app.exception_handler(RequestValidationError)
def request_validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Return a standard JSON payload for request validation failures.

    Args:
        request: Incoming request instance.
        exc: Raised validation exception.

    Returns:
        JSON response using the shared error schema.
    """
    logger.warning(
        "Validation error for %s %s: %s",
        request.method,
        request.url.path,
        exc.errors(),
    )
    payload = ErrorResponse(
        detail="Request validation failed.",
        error_code="validation_error",
    )
    return JSONResponse(status_code=422, content=payload.model_dump())


@app.exception_handler(Exception)
def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a standard JSON payload for unexpected exceptions.

    Args:
        request: Incoming request instance.
        exc: Unhandled application exception.

    Returns:
        JSON response using the shared error schema.
    """
    logger.exception(
        "Unhandled exception for %s %s.",
        request.method,
        request.url.path,
        exc_info=exc,
    )
    payload = ErrorResponse(
        detail="Internal server error.",
        error_code="internal_server_error",
    )
    return JSONResponse(status_code=500, content=payload.model_dump())


@app.get("/health", tags=["health"], response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Return a basic health response for uptime monitoring.

    Returns:
        Minimal service metadata used by health checks.
    """
    logger.debug("Health check requested.")
    return HealthResponse(status="ok", environment=settings.app_env)
