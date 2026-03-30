"""Tests for the FastAPI application entry point."""

from fastapi.testclient import TestClient
from starlette.requests import Request

from app.main import app, unhandled_exception_handler


def test_health_endpoint_returns_service_status() -> None:
    """Ensure the health endpoint returns the expected payload."""
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_unhandled_exception_handler_returns_standard_error_payload() -> None:
    """Ensure unexpected exceptions are serialized with the shared schema."""
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/api/v1/admin/metrics",
            "headers": [],
        }
    )

    response = unhandled_exception_handler(request, RuntimeError("synthetic failure"))

    assert response.status_code == 500
    assert response.body == (
        b'{"detail":"Internal server error.","error_code":"internal_server_error"}'
    )
