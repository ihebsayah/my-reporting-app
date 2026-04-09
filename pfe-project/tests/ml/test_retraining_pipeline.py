"""Tests for the monthly RF retraining pipeline."""

import json
from pathlib import Path

import pytest

from app.retraining.pipeline import RFRetrainingPipeline, RetrainingResult


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

BASE_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "annotation"
    / "rf_training_records.jsonl"
)


def _write_feedback(tmp_path: Path, entries: list) -> Path:
    """Write a feedback JSONL file and return its path."""
    p = tmp_path / "feedback.jsonl"
    with p.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry) + "\n")
    return p


# ──────────────────────────────────────────────────────────────────────────────
# RetrainingPipeline.run() — base only (no feedback)
# ──────────────────────────────────────────────────────────────────────────────


def test_retrain_base_only_returns_result(tmp_path: Path) -> None:
    """run() without feedback must return a complete RetrainingResult."""
    pipeline = RFRetrainingPipeline(use_bert=False)
    result = pipeline.run(
        base_jsonl_path=BASE_FIXTURE,
        output_dir=tmp_path,
        feedback_path=tmp_path / "no_feedback.jsonl",  # does not exist
        n_estimators=5,
    )

    assert isinstance(result, RetrainingResult)
    assert result.base_records > 0
    assert result.feedback_records == 0
    assert result.feedback_file_exists is False
    assert result.total_records == result.base_records


def test_retrain_saves_versioned_joblib(tmp_path: Path) -> None:
    """run() must save a versioned .joblib model file."""
    pipeline = RFRetrainingPipeline(use_bert=False)
    result = pipeline.run(
        base_jsonl_path=BASE_FIXTURE,
        output_dir=tmp_path,
        n_estimators=5,
    )

    model_path = Path(result.model_path)
    assert model_path.exists()
    assert model_path.suffix == ".joblib"
    assert "rf_confidence_v" in model_path.name


def test_retrain_result_accuracy_is_valid(tmp_path: Path) -> None:
    """Training accuracy reported must be between 0 and 1."""
    pipeline = RFRetrainingPipeline(use_bert=False)
    result = pipeline.run(
        base_jsonl_path=BASE_FIXTURE,
        output_dir=tmp_path,
        n_estimators=5,
    )
    assert 0.0 <= result.training_result.accuracy <= 1.0


def test_retrain_missing_base_file_raises(tmp_path: Path) -> None:
    """run() must raise FileNotFoundError when base file is missing."""
    pipeline = RFRetrainingPipeline(use_bert=False)
    with pytest.raises(FileNotFoundError, match="not found"):
        pipeline.run(
            base_jsonl_path=tmp_path / "missing.jsonl",
            output_dir=tmp_path,
            n_estimators=5,
        )


# ──────────────────────────────────────────────────────────────────────────────
# RetrainingPipeline.run() — with feedback
# ──────────────────────────────────────────────────────────────────────────────


def test_retrain_merges_feedback_records(tmp_path: Path) -> None:
    """run() with feedback must include feedback records in total_records."""
    feedback = [
        {
            "document_id": "doc-fb-001",
            "field_name": "INVOICE_ID",
            "correct_value": "INV-FB-001",
            "original_value": "INV-FB-00",
            "original_decision": "review",
            "recorded_at": "2026-04-01T10:00:00Z",
        },
        {
            "document_id": "doc-fb-002",
            "field_name": "TOTAL_AMOUNT",
            "correct_value": "$500.00",
            "original_value": None,
            "original_decision": None,
            "recorded_at": "2026-04-02T10:00:00Z",
        },
    ]
    feedback_path = _write_feedback(tmp_path, feedback)

    pipeline = RFRetrainingPipeline(use_bert=False)
    result = pipeline.run(
        base_jsonl_path=BASE_FIXTURE,
        output_dir=tmp_path,
        feedback_path=feedback_path,
        n_estimators=5,
    )

    assert result.feedback_file_exists is True
    assert result.feedback_records > 0
    assert result.total_records == result.base_records + result.feedback_records


def test_retrain_feedback_with_different_original_adds_negative(tmp_path: Path) -> None:
    """Feedback with a different original_value must produce both TP and FP records."""
    feedback = [
        {
            "document_id": "doc-x",
            "field_name": "INVOICE_ID",
            "correct_value": "INV-CORRECT",
            "original_value": "INV-WRONG",  # different → adds negative record
            "recorded_at": "2026-04-01T10:00:00Z",
        }
    ]
    feedback_path = _write_feedback(tmp_path, feedback)

    pipeline = RFRetrainingPipeline(use_bert=False)
    result = pipeline.run(
        base_jsonl_path=BASE_FIXTURE,
        output_dir=tmp_path,
        feedback_path=feedback_path,
        n_estimators=5,
    )
    # One correct + one incorrect = 2 feedback records for this entry.
    assert result.feedback_records >= 2


# ──────────────────────────────────────────────────────────────────────────────
# Hot-swap integration
# ──────────────────────────────────────────────────────────────────────────────


def test_retrain_hot_swaps_engine(tmp_path: Path) -> None:
    """run() with engine= must call switch_to_rf_model on the engine."""
    from app.ml.ner_extractor import RegexSpacyEnsembleExtractor
    from app.pipeline.decision_engine import SequentialExtractionDecisionEngine

    engine = SequentialExtractionDecisionEngine(
        extractor=RegexSpacyEnsembleExtractor(),
    )
    assert engine._active_scorer == "heuristic"

    pipeline = RFRetrainingPipeline(use_bert=False)
    pipeline.run(
        base_jsonl_path=BASE_FIXTURE,
        output_dir=tmp_path,
        n_estimators=5,
        engine=engine,
    )

    assert engine._active_scorer == "rf"


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────


def test_load_base_records_parses_fixture() -> None:
    """_load_base_records must parse the sample JSONL fixture correctly."""
    records = RFRetrainingPipeline._load_base_records(BASE_FIXTURE)
    assert len(records) >= 2
    labels = {r.entity.label for r in records}
    assert "INVOICE_ID" in labels


def test_load_feedback_records_skips_empty_lines(tmp_path: Path) -> None:
    """_load_feedback_records must silently skip empty lines."""
    p = tmp_path / "feedback.jsonl"
    p.write_text(
        "\n"
        + json.dumps({"field_name": "INVOICE_ID", "correct_value": "INV-1", "document_id": "d"})
        + "\n\n",
        encoding="utf-8",
    )
    records = RFRetrainingPipeline._load_feedback_records(p, BASE_FIXTURE)
    assert len(records) >= 1


def test_merge_records_concatenates_both_lists() -> None:
    """_merge_records must return the union of base and feedback lists."""
    from app.ml.ner_extractor import ExtractedEntity
    from app.ml.rf_confidence_model import TrainingRecord

    def _rec(label: str, val: str) -> TrainingRecord:
        return TrainingRecord(
            entity=ExtractedEntity(0, len(val), val, label, ("regex",), 0.9),
            document_text=val,
            is_correct=True,
        )

    base = [_rec("INVOICE_ID", "INV-1"), _rec("INVOICE_DATE", "2026-01-15")]
    feedback = [_rec("TOTAL_AMOUNT", "$100")]
    merged = RFRetrainingPipeline._merge_records(base, feedback)
    assert len(merged) == 3
