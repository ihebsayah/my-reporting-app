"""Tests for the ML CLI."""

import json
from pathlib import Path
from typing import Any

from app.ml import cli


def test_validate_ner_data_cli_reports_example_count(tmp_path: Path, capsys: Any) -> None:
    """Ensure the validation CLI reports the number of parsed examples."""
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

    exit_code = cli.main(["validate-ner-data", "--input-file", str(input_path)])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["examples"] == 1
    assert payload["errors"] == []


def test_train_ner_cli_outputs_metadata(monkeypatch: Any, tmp_path: Path, capsys: Any) -> None:
    """Ensure the train-ner CLI prints trainer metadata."""
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

    def _fake_train(self: Any, input_path: Path, output_dir: Path, iterations: int) -> dict:
        return {
            "examples": 1,
            "labels": ["INVOICE_ID"],
            "iterations": 5,
            "output_dir": str(output_dir),
            "losses": [],
        }

    monkeypatch.setattr(cli.SpacyNERTrainer, "train_from_jsonl", _fake_train)

    output_dir = tmp_path / "model"
    exit_code = cli.main(
        [
            "train-ner",
            "--input-file",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--iterations",
            "5",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["examples"] == 1
    assert payload["output_dir"] == str(output_dir)


def test_split_ner_data_cli_writes_two_files(tmp_path: Path, capsys: Any) -> None:
    """Ensure split-ner-data saves train and validation JSONL files."""
    input_path = tmp_path / "train.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps({"document_id": f"doc-{idx}", "text": str(idx), "entities": []})
                for idx in range(4)
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    train_path = tmp_path / "train_split.jsonl"
    validation_path = tmp_path / "validation_split.jsonl"

    exit_code = cli.main(
        [
            "split-ner-data",
            "--input-file",
            str(input_path),
            "--train-output-file",
            str(train_path),
            "--validation-output-file",
            str(validation_path),
            "--validation-ratio",
            "0.25",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert train_path.exists()
    assert validation_path.exists()
    assert payload["train_examples"] == 3
    assert payload["validation_examples"] == 1


def test_evaluate_ner_data_cli_outputs_metrics(tmp_path: Path, capsys: Any) -> None:
    """Ensure evaluate-ner-data reports entity-level metrics."""
    gold_path = tmp_path / "gold.jsonl"
    predicted_path = tmp_path / "predicted.jsonl"
    gold_path.write_text(
        json.dumps(
            {
                "document_id": "doc-1",
                "text": "Invoice INV-001 total $12.00",
                "entities": [[8, 15, "INVOICE_ID"], [22, 28, "TOTAL_AMOUNT"]],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    predicted_path.write_text(
        json.dumps(
            {
                "document_id": "doc-1",
                "text": "Invoice INV-001 total $12.00",
                "entities": [[8, 15, "INVOICE_ID"]],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = cli.main(
        [
            "evaluate-ner-data",
            "--gold-file",
            str(gold_path),
            "--predicted-file",
            str(predicted_path),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["true_positives"] == 1
    assert payload["false_negatives"] == 1
    assert round(payload["f1_score"], 3) == 0.667
    assert payload["per_label"]["INVOICE_ID"]["true_positives"] == 1
    assert payload["per_label"]["TOTAL_AMOUNT"]["false_negatives"] == 1


def test_extract_entities_cli_outputs_entities(capsys: Any) -> None:
    """Ensure extract-entities returns ensemble entities for raw text."""
    exit_code = cli.main(
        [
            "extract-entities",
            "--text",
            "Invoice INV-2026-001\nTotal: $12.00",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["entities"]
    assert any(entity["label"] == "INVOICE_ID" for entity in payload["entities"])


def test_run_pipeline_cli_outputs_field_decisions(capsys: Any) -> None:
    """Ensure run-pipeline returns field decisions and an overall decision."""
    exit_code = cli.main(
        [
            "run-pipeline",
            "--text",
            "Invoice INV-2026-001\nTotal: $12.00",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["overall_decision"] in {"auto", "review", "reject"}
    assert payload["fields"]
    assert "confidence_factors" in payload["fields"][0]


def test_run_pipeline_batch_cli_outputs_metrics(capsys: Any) -> None:
    """Ensure run-pipeline-batch reports document and field counters."""
    exit_code = cli.main(
        [
            "run-pipeline-batch",
            "--input-dir",
            "docs/source_documents",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert len(payload["documents"]) == 3
    assert payload["metrics"]["document_count"] == 3
    assert "overall_decisions" in payload["metrics"]


def test_build_kpi_report_cli_outputs_summary(capsys: Any) -> None:
    """Ensure build-kpi-report returns reusable KPI summary data."""
    exit_code = cli.main(
        [
            "build-kpi-report",
            "--input-dir",
            "docs/source_documents",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["document_count"] == 3
    assert "average_field_confidence" in payload
    assert payload["field_kpis"]
