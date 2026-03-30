"""Tests for spaCy-oriented training export helpers."""

import json
from pathlib import Path

from app.annotation.exporter import (
    AnnotatedDocument,
    SpanAnnotation,
    convert_to_spacy_examples,
    load_training_export,
    save_spacy_jsonl,
    save_training_export,
)


def test_convert_to_spacy_examples_builds_entity_triplets() -> None:
    """Ensure normalized annotations become spaCy-style entity offsets."""
    documents = [
        AnnotatedDocument(
            document_id="doc-001",
            text="Invoice INV-001 total $12.00",
            annotations=[
                SpanAnnotation(8, 15, "INV-001", "INVOICE_ID"),
                SpanAnnotation(22, 28, "$12.00", "TOTAL_AMOUNT"),
            ],
        )
    ]

    examples = convert_to_spacy_examples(documents)

    assert examples[0]["document_id"] == "doc-001"
    assert examples[0]["entities"] == [[8, 15, "INVOICE_ID"], [22, 28, "TOTAL_AMOUNT"]]


def test_save_and_load_training_export_round_trip(tmp_path: Path) -> None:
    """Ensure normalized training exports can be reloaded for downstream conversion."""
    documents = [
        AnnotatedDocument(
            document_id="doc-002",
            text="Invoice INV-002",
            annotations=[SpanAnnotation(8, 15, "INV-002", "INVOICE_ID")],
        )
    ]
    training_path = tmp_path / "training.json"

    save_training_export(documents, training_path)
    loaded_documents = load_training_export(training_path)

    assert loaded_documents[0].document_id == "doc-002"
    assert loaded_documents[0].annotations[0].label == "INVOICE_ID"


def test_save_spacy_jsonl_writes_one_json_object_per_line(tmp_path: Path) -> None:
    """Ensure spaCy JSONL export is saved as newline-delimited JSON."""
    documents = [
        AnnotatedDocument(
            document_id="doc-003",
            text="Vendor Acme",
            annotations=[SpanAnnotation(7, 11, "Acme", "VENDOR_NAME")],
        )
    ]
    output_path = tmp_path / "spacy.jsonl"

    save_spacy_jsonl(documents, output_path)
    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    payload = json.loads(lines[0])

    assert len(lines) == 1
    assert payload["entities"] == [[7, 11, "VENDOR_NAME"]]
