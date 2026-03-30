"""Tests for annotation bootstrap artifact generation and export helpers."""

import json
from pathlib import Path

from app.annotation.bootstrap import build_annotation_bootstrap_artifacts
from app.annotation.exporter import AnnotatedDocument, SpanAnnotation, export_for_training
from app.config import Settings


def test_bootstrap_artifacts_are_written(tmp_path: Path) -> None:
    """Ensure bootstrap files are generated with project-specific content."""
    settings = Settings(LABEL_STUDIO_PROJECT_TITLE="Accounts Payable Invoices")

    paths = build_annotation_bootstrap_artifacts(tmp_path, settings=settings)

    schema_payload = json.loads(Path(paths["schema"]).read_text(encoding="utf-8"))
    config_payload = Path(paths["label_config"]).read_text(encoding="utf-8")
    guidelines_payload = Path(paths["guidelines"]).read_text(encoding="utf-8")

    assert len(schema_payload) >= 5
    assert "Accounts Payable Invoices" in config_payload
    assert "Accounts Payable Invoices" in guidelines_payload


def test_export_for_training_preserves_annotation_shape() -> None:
    """Ensure exported training records preserve document and span metadata."""
    document = AnnotatedDocument(
        document_id="doc-001",
        text="Invoice INV-2026-001 total is $12.00",
        annotations=[
            SpanAnnotation(start=8, end=20, text="INV-2026-001", label="INVOICE_ID"),
            SpanAnnotation(start=30, end=36, text="$12.00", label="TOTAL_AMOUNT"),
        ],
    )

    exported = export_for_training([document])

    assert exported[0]["document_id"] == "doc-001"
    assert exported[0]["annotations"][0]["label"] == "INVOICE_ID"
    assert exported[0]["annotations"][1]["text"] == "$12.00"
