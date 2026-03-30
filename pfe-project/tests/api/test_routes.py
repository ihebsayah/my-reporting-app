"""Tests for the FastAPI API routes."""

from fastapi.testclient import TestClient

from app.main import app
client = TestClient(app)


def test_extract_endpoint_returns_entities() -> None:
    """Ensure the extraction endpoint returns detected entities."""
    response = client.post(
        "/api/v1/extract",
        json={"text": "Invoice INV-2026-001\nTotal: $12.00"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["entities"]
    assert any(entity["label"] == "INVOICE_ID" for entity in payload["entities"])


def test_pipeline_endpoint_returns_field_decisions() -> None:
    """Ensure the pipeline endpoint returns overall and field-level decisions."""
    response = client.post(
        "/api/v1/pipeline/run",
        json={"text": "Invoice INV-2026-001\nVendor: Acme Supplies LLC\nTotal: $12.00"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["overall_decision"] in {"auto", "review", "reject"}
    assert payload["fields"]
    assert "confidence_factors" in payload["fields"][0]


def test_batch_pipeline_endpoint_returns_metrics() -> None:
    """Ensure the batch pipeline endpoint returns aggregated metrics."""
    response = client.post(
        "/api/v1/pipeline/batch",
        json={
            "texts": [
                "Invoice INV-2026-001\nTotal: $12.00",
                "Invoice INV-2026-002\nVendor: Globex SARL\nTotal: 480.500 TND",
            ],
            "document_ids": ["doc-1", "doc-2"],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["documents"]) == 2
    assert payload["metrics"]["document_count"] == 2
    assert "overall_decisions" in payload["metrics"]


def test_async_batch_submission_and_polling_return_job_result() -> None:
    """Ensure async batch submission returns a pollable job with results."""
    submit_response = client.post(
        "/api/v1/pipeline/batch/submit",
        json={
            "texts": [
                "Invoice INV-2026-001\nTotal: $12.00",
                "Invoice INV-2026-002\nVendor: Globex SARL\nTotal: 480.500 TND",
            ],
            "document_ids": ["doc-1", "doc-2"],
        },
    )

    assert submit_response.status_code == 200
    submit_payload = submit_response.json()
    assert submit_payload["job_id"]
    assert submit_payload["status"] in {"pending", "running", "completed"}

    status_response = client.get(
        f"/api/v1/pipeline/batch/jobs/{submit_payload['job_id']}"
    )

    assert status_response.status_code == 200
    status_payload = status_response.json()
    assert status_payload["job_id"] == submit_payload["job_id"]
    assert status_payload["status"] == "completed"
    assert status_payload["result"] is not None
    assert status_payload["result"]["metrics"]["document_count"] == 2


def test_kpi_report_endpoint_returns_summary() -> None:
    """Ensure the KPI endpoint returns reusable KPI metrics."""
    response = client.post(
        "/api/v1/kpi/report",
        json={
            "texts": [
                "Invoice INV-2026-001\nVendor: Acme Supplies LLC\nTotal: $12.00",
                "Invoice INV-2026-002\nTotal: 480.500 TND",
            ],
            "document_ids": ["doc-1", "doc-2"],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["document_count"] == 2
    assert "average_field_confidence" in payload
    assert payload["field_kpis"]


def test_admin_status_endpoint_returns_versioned_service_metadata() -> None:
    """Ensure the admin status endpoint exposes safe runtime metadata."""
    response = client.get("/api/v1/admin/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["app_name"]
    assert "field_thresholds" in payload
    assert "default_thresholds" in payload


def test_admin_model_endpoint_returns_model_metadata() -> None:
    """Ensure the admin model endpoint exposes model availability details."""
    response = client.get("/api/v1/admin/model")

    assert response.status_code == 200
    payload = response.json()
    assert "ner_model_path" in payload
    assert "spacy_available" in payload
    assert "train_iterations" in payload


def test_admin_metrics_endpoint_returns_kpi_summary() -> None:
    """Ensure the admin metrics endpoint builds KPI data from source documents."""
    response = client.get("/api/v1/admin/metrics")

    assert response.status_code == 200
    payload = response.json()
    assert payload["document_count"] >= 1
    assert "field_kpis" in payload


def test_batch_pipeline_endpoint_returns_standard_error_payload() -> None:
    """Ensure handled HTTP errors follow the shared error schema."""
    response = client.post(
        "/api/v1/pipeline/batch",
        json={"texts": ["Invoice INV-2026-001"], "document_ids": ["doc-1", "doc-2"]},
    )

    assert response.status_code == 400
    payload = response.json()
    assert payload["detail"] == "document_ids length must match texts length."
    assert payload["error_code"] == "http_400"


def test_async_batch_status_endpoint_returns_not_found_error() -> None:
    """Ensure missing async jobs return the shared not-found payload."""
    response = client.get("/api/v1/pipeline/batch/jobs/missing-job")

    assert response.status_code == 404
    payload = response.json()
    assert payload["detail"] == "Batch job was not found."
    assert payload["error_code"] == "http_404"


def test_extract_endpoint_returns_standard_validation_error_payload() -> None:
    """Ensure request validation failures use the shared error schema."""
    response = client.post("/api/v1/extract", json={"text": ""})

    assert response.status_code == 422
    payload = response.json()
    assert payload["detail"] == "Request validation failed."
    assert payload["error_code"] == "validation_error"
