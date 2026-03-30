"""Tests for NER splitting and evaluation helpers."""

import json
from pathlib import Path

from app.ml.ner_trainer import SpacyNERTrainer


def _write_jsonl(path: Path, rows: list) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_split_examples_creates_train_and_validation_sets(tmp_path: Path) -> None:
    """Ensure splitting produces non-empty deterministic partitions."""
    input_path = tmp_path / "train.jsonl"
    _write_jsonl(
        input_path,
        [
            {"document_id": "doc-1", "text": "A", "entities": []},
            {"document_id": "doc-2", "text": "B", "entities": []},
            {"document_id": "doc-3", "text": "C", "entities": []},
            {"document_id": "doc-4", "text": "D", "entities": []},
        ],
    )
    trainer = SpacyNERTrainer()
    examples = trainer.load_examples(input_path)

    train_examples, validation_examples = trainer.split_examples(
        examples,
        validation_ratio=0.25,
        random_seed=7,
    )

    assert len(train_examples) == 3
    assert len(validation_examples) == 1
    assert train_examples[0].document_id != validation_examples[0].document_id


def test_evaluate_predictions_computes_entity_level_metrics(tmp_path: Path) -> None:
    """Ensure precision, recall, and F1 are computed from exact span matches."""
    gold_path = tmp_path / "gold.jsonl"
    predicted_path = tmp_path / "predicted.jsonl"
    _write_jsonl(
        gold_path,
        [
            {
                "document_id": "doc-1",
                "text": "Invoice INV-001 total $12.00",
                "entities": [[8, 15, "INVOICE_ID"], [22, 28, "TOTAL_AMOUNT"]],
            }
        ],
    )
    _write_jsonl(
        predicted_path,
        [
            {
                "document_id": "doc-1",
                "text": "Invoice INV-001 total $12.00",
                "entities": [[8, 15, "INVOICE_ID"]],
            }
        ],
    )
    trainer = SpacyNERTrainer()

    report = trainer.evaluate_predictions(
        trainer.load_examples(gold_path),
        trainer.load_examples(predicted_path),
    )

    assert report.overall.true_positives == 1
    assert report.overall.false_positives == 0
    assert report.overall.false_negatives == 1
    assert round(report.overall.precision, 3) == 1.0
    assert round(report.overall.recall, 3) == 0.5
    assert round(report.overall.f1_score, 3) == 0.667
    assert report.per_label["INVOICE_ID"].true_positives == 1
    assert report.per_label["TOTAL_AMOUNT"].false_negatives == 1
