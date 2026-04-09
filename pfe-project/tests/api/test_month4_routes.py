"""Tests for Month 4 API routes: /monitoring/drift and /admin/retrain."""

import json

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# ──────────────────────────────────────────────────────────────────────────────
# GET /monitoring/drift
# ──────────────────────────────────────────────────────────────────────────────


def test_drift_endpoint_returns_valid_json() -> None:
    """GET /monitoring/drift must return 200 with a complete DriftResponse."""
    response = client.get("/api/v1/monitoring/drift")

    assert response.status_code == 200
    payload = response.json()
    assert "drift_detected" in payload
    assert isinstance(payload["drift_detected"], bool)
    assert "triggered_signals" in payload
    assert isinstance(payload["triggered_signals"], list)
    assert "checked_at" in payload
    assert "T" in payload["checked_at"]
    assert "metadata" in payload
    assert payload["metadata"]["processed_at"]


def test_drift_endpoint_reports_insufficient_history_without_data() -> None:
    """GET /monitoring/drift must set insufficient_history=True if DB lacks records."""
    # Use a large window that definitely exceeds the number of stored records.
    response = client.get(
        "/api/v1/monitoring/drift",
        params={"baseline_window": 500, "recent_window": 200},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["insufficient_history"] is True
    assert payload["drift_detected"] is False


def test_drift_endpoint_accepts_custom_thresholds() -> None:
    """GET /monitoring/drift must accept custom auto_threshold and confidence_threshold."""
    response = client.get(
        "/api/v1/monitoring/drift",
        params={
            "baseline_window": 5,
            "recent_window": 5,
            "auto_threshold": 0.20,
            "confidence_threshold": 0.10,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["auto_rate_threshold"] == pytest.approx(0.20)
    assert payload["confidence_threshold"] == pytest.approx(0.10)


def test_drift_endpoint_no_drift_fields_when_insufficient() -> None:
    """baseline and recent fields should be None when history is insufficient."""
    response = client.get(
        "/api/v1/monitoring/drift",
        params={"baseline_window": 500, "recent_window": 200},
    )
    payload = response.json()
    assert payload["baseline"] is None
    assert payload["recent"] is None


# ──────────────────────────────────────────────────────────────────────────────
# POST /admin/retrain
# ──────────────────────────────────────────────────────────────────────────────


def test_admin_retrain_returns_success_with_model_version() -> None:
    """POST /admin/retrain must succeed and return model_version and accuracy."""
    response = client.post(
        "/api/v1/admin/retrain",
        params={"n_estimators": 5},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["model_version"] is not None
    assert "rf_confidence_v" in payload["model_version"]
    assert 0.0 <= payload["accuracy"] <= 1.0
    assert payload["total_records"] > 0
    assert payload["metadata"]["processed_at"]


def test_admin_retrain_reports_record_counts() -> None:
    """POST /admin/retrain must include total_records and feedback_records."""
    response = client.post("/api/v1/admin/retrain", params={"n_estimators": 5})

    payload = response.json()
    assert payload["success"] is True
    assert isinstance(payload["total_records"], int)
    assert isinstance(payload["feedback_records"], int)
    assert payload["total_records"] >= payload["feedback_records"]


def test_admin_retrain_hot_swaps_engine() -> None:
    """After POST /admin/retrain, GET /pipeline/run should use scorer=rf."""
    response = client.post("/api/v1/admin/retrain", params={"n_estimators": 5})
    assert response.json()["success"] is True

    pipeline_response = client.post(
        "/api/v1/pipeline/run",
        json={"text": "Invoice INV-2026-001 Total: $1,200.00"},
    )
    assert pipeline_response.status_code == 200
    assert pipeline_response.json()["scorer"] == "rf"


def test_admin_retrain_returns_error_on_missing_base_file() -> None:
    """POST /admin/retrain must return success=False when base file is absent."""
    # We can't easily remove the real fixture, but we can monkey-patch.
    import unittest.mock as mock
    from app.retraining.pipeline import RFRetrainingPipeline

    with mock.patch.object(
        RFRetrainingPipeline,
        "run",
        side_effect=FileNotFoundError("Base training file not found: missing.jsonl"),
    ):
        response = client.post("/api/v1/admin/retrain", params={"n_estimators": 5})

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is False
    assert payload["error"] is not None
    assert "not found" in payload["error"].lower()
