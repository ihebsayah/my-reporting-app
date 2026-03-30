"""Tests for the spaCy NER training helper."""

import json
from pathlib import Path
from typing import Any

import pytest

from app.ml.ner_trainer import NERTrainingError, SpacyNERTrainer


def test_load_examples_reads_spacy_jsonl(tmp_path: Path) -> None:
    """Ensure spaCy JSONL examples are parsed into dataclasses."""
    input_path = tmp_path / "train.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "document_id": "doc-001",
                "text": "Invoice INV-001",
                "entities": [[8, 15, "INVOICE_ID"]],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    examples = SpacyNERTrainer().load_examples(input_path)

    assert len(examples) == 1
    assert examples[0].document_id == "doc-001"
    assert examples[0].entities[0] == (8, 15, "INVOICE_ID")


def test_validate_examples_detects_invalid_offsets() -> None:
    """Ensure invalid entity offsets are reported."""
    trainer = SpacyNERTrainer()
    sample_path = Path(__file__).resolve().parents[2] / "docs" / "annotation" / "spacy_train.jsonl"
    examples = trainer.load_examples(
        sample_path
    )
    broken_example = examples[0].__class__(
        document_id=examples[0].document_id,
        text=examples[0].text,
        entities=[(30, 10, "BROKEN_LABEL")],
    )

    errors = trainer.validate_examples([broken_example])

    assert errors
    assert "invalid entity offsets" in errors[0]


def test_train_from_jsonl_without_spacy_raises_clear_error(monkeypatch: Any, tmp_path: Path) -> None:
    """Ensure training fails clearly when spaCy is unavailable."""
    input_path = tmp_path / "train.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "document_id": "doc-001",
                "text": "Invoice INV-001",
                "entities": [[8, 15, "INVOICE_ID"]],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def _fake_import(name: str) -> Any:
        raise ImportError("spacy missing")

    monkeypatch.setattr("app.ml.ner_trainer.importlib.import_module", _fake_import)

    with pytest.raises(NERTrainingError) as exc_info:
        SpacyNERTrainer().train_from_jsonl(
            input_path=input_path,
            output_dir=tmp_path / "model",
        )

    assert "spaCy is required" in str(exc_info.value)
