"""Tests for the Month 4 CLI commands: retrain-rf and check-drift."""

import json
from pathlib import Path

import pytest

from app.ml.cli import main


BASE_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "annotation"
    / "rf_training_records.jsonl"
)


# ──────────────────────────────────────────────────────────────────────────────
# retrain-rf
# ──────────────────────────────────────────────────────────────────────────────


def test_retrain_rf_cli_outputs_model_version(tmp_path: Path, capsys) -> None:
    """retrain-rf must print a JSON payload with model_version and accuracy."""
    exit_code = main(
        [
            "retrain-rf",
            "--input-file", str(BASE_FIXTURE),
            "--output-dir", str(tmp_path),
            "--n-estimators", "5",
            "--no-bert",
        ]
    )
    assert exit_code == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert "model_version" in payload
    assert "rf_confidence_v" in payload["model_version"]
    assert 0.0 <= payload["accuracy"] <= 1.0


def test_retrain_rf_cli_outputs_record_counts(tmp_path: Path, capsys) -> None:
    """retrain-rf must report base_records, feedback_records, and total_records."""
    main(
        [
            "retrain-rf",
            "--input-file", str(BASE_FIXTURE),
            "--output-dir", str(tmp_path),
            "--n-estimators", "5",
            "--no-bert",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["base_records"] > 0
    assert payload["feedback_records"] == 0        # no feedback file provided
    assert payload["total_records"] == payload["base_records"]


def test_retrain_rf_cli_saves_joblib_file(tmp_path: Path, capsys) -> None:
    """retrain-rf must save a .joblib file at the reported model_path."""
    main(
        [
            "retrain-rf",
            "--input-file", str(BASE_FIXTURE),
            "--output-dir", str(tmp_path),
            "--n-estimators", "5",
            "--no-bert",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert Path(payload["model_path"]).exists()


def test_retrain_rf_cli_with_feedback_file(tmp_path: Path, capsys) -> None:
    """retrain-rf must load feedback records when --feedback-file is specified."""
    feedback_path = tmp_path / "feedback.jsonl"
    feedback_path.write_text(
        json.dumps(
            {
                "document_id": "fb-1",
                "field_name": "INVOICE_ID",
                "correct_value": "INV-999",
                "original_value": "INV-99",
                "recorded_at": "2026-04-01T10:00:00Z",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    main(
        [
            "retrain-rf",
            "--input-file", str(BASE_FIXTURE),
            "--output-dir", str(tmp_path),
            "--feedback-file", str(feedback_path),
            "--n-estimators", "5",
            "--no-bert",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["feedback_file_exists"] is True
    assert payload["feedback_records"] >= 1
    assert payload["total_records"] > payload["base_records"]


def test_retrain_rf_cli_missing_input_raises(tmp_path: Path) -> None:
    """retrain-rf must propagate FileNotFoundError for missing --input-file."""
    with pytest.raises(FileNotFoundError):
        main(
            [
                "retrain-rf",
                "--input-file", str(tmp_path / "missing.jsonl"),
                "--output-dir", str(tmp_path),
                "--n-estimators", "5",
                "--no-bert",
            ]
        )


# ──────────────────────────────────────────────────────────────────────────────
# check-drift
# ──────────────────────────────────────────────────────────────────────────────


def test_check_drift_cli_returns_json(capsys) -> None:
    """check-drift must output valid JSON with drift_detected and checked_at."""
    exit_code = main(
        [
            "check-drift",
            "--baseline-window", "5",
            "--recent-window", "5",
        ]
    )
    assert exit_code == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert "drift_detected" in payload
    assert isinstance(payload["drift_detected"], bool)
    assert "checked_at" in payload
    assert "T" in payload["checked_at"]


def test_check_drift_cli_reports_baseline_and_recent_stats(capsys) -> None:
    """check-drift payload must include baseline and recent window statistics."""
    main(["check-drift", "--baseline-window", "5", "--recent-window", "5"])
    payload = json.loads(capsys.readouterr().out)
    assert "baseline" in payload
    assert "recent" in payload
    assert "window_size" in payload["baseline"]
    assert "auto_rate" in payload["recent"]


def test_check_drift_cli_no_drift_without_enough_records(capsys) -> None:
    """check-drift should report no drift when DB has insufficient records."""
    main(
        [
            "check-drift",
            "--baseline-window", "500",  # far more than any test DB has
            "--recent-window", "200",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    # With insufficient records the detector returns drift_detected=False
    assert payload["drift_detected"] is False
    assert payload["triggered_signals"] == []


def test_check_drift_cli_respects_custom_thresholds(capsys) -> None:
    """check-drift must pass custom --auto-threshold and --confidence-threshold through."""
    main(
        [
            "check-drift",
            "--baseline-window", "5",
            "--recent-window", "5",
            "--auto-threshold", "0.20",
            "--confidence-threshold", "0.10",
        ]
    )
    # No assertion on drift_detected (depends on DB state);
    # just ensure it parses and outputs clean JSON.
    payload = json.loads(capsys.readouterr().out)
    assert "drift_detected" in payload
