"""Tests for the Month 3 API routes: feedback, NER KPI, storage KPI, extraction history."""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

INVOICE_TEXT = "Invoice INV-2026-001 dated 2026-01-15. Vendor: Acme SARL. Total: $1,200.00"


# ──────────────────────────────────────────────────────────────────────────────
# /pipeline/run — scorer field (Month 3 addition)
# ──────────────────────────────────────────────────────────────────────────────


def test_pipeline_run_response_includes_scorer_field() -> None:
    """POST /pipeline/run must include 'scorer' in the response."""
    response = client.post("/api/v1/pipeline/run", json={"text": INVOICE_TEXT})

    assert response.status_code == 200
    payload = response.json()
    assert "scorer" in payload
    assert payload["scorer"] in {"heuristic", "rf"}


# ──────────────────────────────────────────────────────────────────────────────
# GET /admin/model — RF availability fields
# ──────────────────────────────────────────────────────────────────────────────


def test_admin_model_includes_rf_availability() -> None:
    """GET /admin/model must include rf_model_available and rf_model_version."""
    response = client.get("/api/v1/admin/model")

    assert response.status_code == 200
    payload = response.json()
    assert "rf_model_available" in payload
    assert isinstance(payload["rf_model_available"], bool)
    # rf_model_version is either None or a string filename
    assert "rf_model_version" in payload


# ──────────────────────────────────────────────────────────────────────────────
# POST /feedback
# ──────────────────────────────────────────────────────────────────────────────


def test_feedback_endpoint_records_correction() -> None:
    """POST /feedback must accept a correction and confirm it was recorded."""
    response = client.post(
        "/api/v1/feedback",
        json={
            "document_id": "doc-feedback-001",
            "field_name": "INVOICE_ID",
            "correct_value": "INV-2026-001",
            "original_value": "INV-2026-00",
            "original_decision": "review",
            "notes": "Missing trailing digit.",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["document_id"] == "doc-feedback-001"
    assert payload["field_name"] == "INVOICE_ID"
    assert payload["correct_value"] == "INV-2026-001"
    assert payload["recorded"] is True
    assert payload["message"]
    assert payload["metadata"]["processed_at"]


def test_feedback_endpoint_accepts_minimal_payload() -> None:
    """POST /feedback must work with only required fields."""
    response = client.post(
        "/api/v1/feedback",
        json={
            "document_id": "doc-feedback-002",
            "field_name": "TOTAL_AMOUNT",
            "correct_value": "$1,200.00",
        },
    )

    assert response.status_code == 200
    assert response.json()["recorded"] is True


def test_feedback_endpoint_rejects_empty_document_id() -> None:
    """POST /feedback must reject an empty document_id with 422."""
    response = client.post(
        "/api/v1/feedback",
        json={
            "document_id": "",
            "field_name": "INVOICE_ID",
            "correct_value": "INV-001",
        },
    )

    assert response.status_code == 422


def test_feedback_endpoint_rejects_missing_correct_value() -> None:
    """POST /feedback must reject a request missing correct_value with 422."""
    response = client.post(
        "/api/v1/feedback",
        json={
            "document_id": "doc-001",
            "field_name": "INVOICE_ID",
        },
    )

    assert response.status_code == 422


# ──────────────────────────────────────────────────────────────────────────────
# POST /kpi/ner
# ──────────────────────────────────────────────────────────────────────────────


def test_ner_kpi_endpoint_returns_report() -> None:
    """POST /kpi/ner must return a full NER KPI report."""
    gold = [
        {"document_id": "d1", "label": "INVOICE_ID", "value": "INV-001"},
        {"document_id": "d1", "label": "TOTAL_AMOUNT", "value": "$1,200.00"},
    ]
    predicted = [
        {"document_id": "d1", "label": "INVOICE_ID", "value": "INV-001"},
        {"document_id": "d1", "label": "TOTAL_AMOUNT", "value": "$1,200.00"},
    ]
    response = client.post(
        "/api/v1/kpi/ner",
        params={"f1_target": 0.85},
        json={"gold": gold, "predicted": predicted},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["overall_f1"] == pytest.approx(1.0, abs=0.01)
    assert payload["meets_target"] is True
    assert payload["per_label"]
    assert payload["metadata"]["processed_at"]


def test_ner_kpi_endpoint_reports_partial_match() -> None:
    """POST /kpi/ner must reflect partial matches in F1 < 1.0."""
    gold = [
        {"document_id": "d1", "label": "INVOICE_ID", "value": "INV-001"},
        {"document_id": "d1", "label": "VENDOR_NAME", "value": "Acme"},
    ]
    predicted = [
        {"document_id": "d1", "label": "INVOICE_ID", "value": "INV-001"},
        # VENDOR_NAME missing → FN
    ]
    response = client.post(
        "/api/v1/kpi/ner",
        params={"f1_target": 0.99},
        json={"gold": gold, "predicted": predicted},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["overall_f1"] < 1.0
    assert payload["meets_target"] is False


def test_ner_kpi_endpoint_handles_empty_inputs() -> None:
    """POST /kpi/ner must handle empty gold and predicted gracefully."""
    response = client.post(
        "/api/v1/kpi/ner",
        json={"gold": [], "predicted": []},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["overall_f1"] == pytest.approx(0.0, abs=0.01)
    assert payload["document_count"] == 0


# ──────────────────────────────────────────────────────────────────────────────
# GET /kpi/storage
# ──────────────────────────────────────────────────────────────────────────────


def test_storage_kpi_endpoint_returns_rates() -> None:
    """GET /kpi/storage must return auto/review/reject rates from the DB."""
    # First persist some pipeline results via the run endpoint.
    client.post("/api/v1/pipeline/run", json={"text": INVOICE_TEXT})

    response = client.get("/api/v1/kpi/storage")

    assert response.status_code == 200
    payload = response.json()
    assert "total_documents" in payload
    assert "auto_rate" in payload
    assert "review_rate" in payload
    assert "reject_rate" in payload
    assert "average_confidence_by_field" in payload
    assert payload["metadata"]["processed_at"]


def test_storage_kpi_rates_sum_to_one() -> None:
    """auto_rate + review_rate + reject_rate must sum to ~1.0 when docs exist."""
    # Ensure at least one persisted result.
    client.post("/api/v1/pipeline/run", json={"text": INVOICE_TEXT})

    response = client.get("/api/v1/kpi/storage")
    payload = response.json()

    if payload["total_documents"] > 0:
        total = payload["auto_rate"] + payload["review_rate"] + payload["reject_rate"]
        assert total == pytest.approx(1.0, abs=0.01)


# ──────────────────────────────────────────────────────────────────────────────
# GET /extractions/{document_id}
# ──────────────────────────────────────────────────────────────────────────────


def test_extraction_history_returns_results_for_document() -> None:
    """GET /extractions/{document_id} must return historical results from DB."""
    # Persist at least one run so the DB has a record.
    client.post("/api/v1/pipeline/run", json={"text": INVOICE_TEXT})

    # The document_id is the first 40 chars with spaces replaced by underscores.
    doc_key = INVOICE_TEXT[:40].replace(" ", "_")
    response = client.get(f"/api/v1/extractions/{doc_key}")

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, list)
    if payload:
        first = payload[0]
        assert "record_id" in first
        assert "overall_decision" in first
        assert "scorer" in first
        assert "fields" in first
        assert "processed_at" in first


def test_extraction_history_returns_empty_list_for_unknown_document() -> None:
    """GET /extractions/{document_id} must return [] for an unknown document."""
    response = client.get("/api/v1/extractions/nonexistent-document-xyz")

    assert response.status_code == 200
    assert response.json() == []


def test_extraction_history_validates_limit() -> None:
    """GET /extractions/{document_id} must reject limit=0 with 400."""
    response = client.get("/api/v1/extractions/doc-001?limit=0")

    assert response.status_code == 400
    payload = response.json()
    assert "limit" in payload["detail"]
