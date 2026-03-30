"""Tests for task building and Label Studio export normalization."""

import json
from pathlib import Path

from app.annotation.exporter import parse_label_studio_export
from app.annotation.task_builder import build_label_studio_tasks, load_documents_from_directory


def test_load_documents_from_directory_supports_txt_json_and_csv() -> None:
    """Ensure text, JSON, and CSV source documents are loaded correctly."""
    fixtures_dir = Path(__file__).resolve().parents[2] / "docs" / "source_documents"

    documents = load_documents_from_directory(fixtures_dir)

    assert len(documents) == 3
    assert documents[0].metadata["file_type"] == "txt"
    assert documents[1].metadata["source_type"] == "json"
    assert documents[2].metadata["file_type"] == "csv"


def test_build_label_studio_tasks_preserves_document_ids() -> None:
    """Ensure task payloads include text and document IDs."""
    fixtures_dir = Path(__file__).resolve().parents[2] / "docs" / "source_documents"
    documents = load_documents_from_directory(fixtures_dir)

    tasks = build_label_studio_tasks(documents)

    assert len(tasks) == 3
    assert tasks[0]["data"]["document_id"] == "invoice_001"
    assert "text" in tasks[1]["data"]
    assert tasks[2]["meta"]["file_type"] == "csv"


def test_parse_label_studio_export_normalizes_span_annotations() -> None:
    """Ensure raw Label Studio exports become training-ready annotated documents."""
    payload = [
        {
            "id": 101,
            "data": {
                "document_id": "invoice-101",
                "text": "Invoice INV-2026-101 total is $12.00",
            },
            "annotations": [
                {
                    "result": [
                        {
                            "type": "labels",
                            "value": {
                                "start": 8,
                                "end": 20,
                                "text": "INV-2026-101",
                                "labels": ["INVOICE_ID"],
                            },
                        },
                        {
                            "type": "labels",
                            "value": {
                                "start": 30,
                                "end": 36,
                                "text": "$12.00",
                                "labels": ["TOTAL_AMOUNT"],
                            },
                        },
                    ]
                }
            ],
        }
    ]

    documents = parse_label_studio_export(payload)

    assert documents[0].document_id == "invoice-101"
    assert documents[0].annotations[0].label == "INVOICE_ID"
    assert documents[0].annotations[1].text == "$12.00"
