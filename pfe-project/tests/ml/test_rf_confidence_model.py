"""Tests for RFConfidenceModel.

Strategy
--------
* scikit-learn and joblib are standard dependencies — mocking them is avoided;
  we train tiny forests (n_estimators=5) to keep tests fast.
* BERT is disabled via ``use_bert=False`` throughout.
* The test suite covers training, prediction, batch prediction, decision routing,
  missing-model guard, load/save round-trip, and CLI helpers.
"""

import json
from pathlib import Path
from typing import List

import pytest

from app.ml.feature_builder import EntityFeatureBuilder
from app.ml.ner_extractor import ExtractedEntity
from app.ml.rf_confidence_model import (
    ConfidencePrediction,
    RFConfidenceModel,
    RFModelNotLoadedError,
    RFModelTrainingError,
    RFTrainingResult,
    TrainingRecord,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

DOC = "Invoice INV-2026-001 dated 2026-01-15. Vendor: Acme SARL. Total: $1,200.00"


def _entity(
    label: str = "INVOICE_ID",
    text: str = "INV-2026-001",
    start: int = 8,
    end: int = 20,
    sources: tuple = ("regex",),
    score: float = 0.85,
) -> ExtractedEntity:
    return ExtractedEntity(
        start=start, end=end, text=text, label=label, sources=sources, score=score
    )


def _correct_record(label: str = "INVOICE_ID") -> TrainingRecord:
    return TrainingRecord(entity=_entity(label=label), document_text=DOC, is_correct=True)


def _incorrect_record(label: str = "INVOICE_ID") -> TrainingRecord:
    return TrainingRecord(
        entity=_entity(label=label, text="?", start=0, end=1, score=0.1),
        document_text=DOC,
        is_correct=False,
    )


def _minimal_records() -> List[TrainingRecord]:
    """Return at least 2 records (1 positive, 1 negative) for training."""
    return [
        _correct_record("INVOICE_ID"),
        _correct_record("INVOICE_DATE"),
        _correct_record("TOTAL_AMOUNT"),
        _incorrect_record("INVOICE_ID"),
        _incorrect_record("INVOICE_DATE"),
    ]


def _trained_model(tmp_path: Path) -> RFConfidenceModel:
    """Return an already-trained RFConfidenceModel (fast, 5 trees, no BERT)."""
    model = RFConfidenceModel(use_bert=False)
    model.train(_minimal_records(), output_dir=tmp_path, n_estimators=5)
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────


def test_train_returns_result_metadata(tmp_path: Path) -> None:
    """train() must return RFTrainingResult with expected fields."""
    model = RFConfidenceModel(use_bert=False)
    result = model.train(_minimal_records(), output_dir=tmp_path, n_estimators=5)

    assert isinstance(result, RFTrainingResult)
    assert result.train_samples == len(_minimal_records())
    assert 0.0 <= result.accuracy <= 1.0
    assert result.feature_count > 0
    assert result.model_version.startswith("rf_confidence_v")
    assert result.trained_at  # ISO timestamp


def test_train_saves_joblib_file(tmp_path: Path) -> None:
    """train() must persist a versioned .joblib file."""
    model = RFConfidenceModel(use_bert=False)
    result = model.train(_minimal_records(), output_dir=tmp_path, n_estimators=5)

    model_path = Path(result.model_path)
    assert model_path.exists()
    assert model_path.suffix == ".joblib"


def test_train_saves_metadata_json(tmp_path: Path) -> None:
    """train() must write a sidecar JSON metadata file."""
    model = RFConfidenceModel(use_bert=False)
    result = model.train(_minimal_records(), output_dir=tmp_path, n_estimators=5)

    metadata_path = Path(result.model_path).with_suffix(".json")
    assert metadata_path.exists()
    payload = json.loads(metadata_path.read_text())
    assert payload["model_version"] == result.model_version
    assert payload["train_samples"] == result.train_samples


def test_train_fewer_than_2_records_raises() -> None:
    """train() with a single record must raise ValueError."""
    model = RFConfidenceModel(use_bert=False)
    with pytest.raises(ValueError, match="2"):
        model.train([_correct_record()], n_estimators=5)


def test_train_includes_all_labels(tmp_path: Path) -> None:
    """RFTrainingResult.labels must include every NER label seen in records."""
    records = [
        _correct_record("INVOICE_ID"),
        _correct_record("TOTAL_AMOUNT"),
        _incorrect_record("VENDOR_NAME"),
    ]
    model = RFConfidenceModel(use_bert=False)
    result = model.train(records, output_dir=tmp_path, n_estimators=5)
    assert set(result.labels) >= {"INVOICE_ID", "TOTAL_AMOUNT", "VENDOR_NAME"}


# ──────────────────────────────────────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────────────────────────────────────


def test_predict_returns_confidence_prediction(tmp_path: Path) -> None:
    """predict() must return a ConfidencePrediction with all fields set."""
    model = _trained_model(tmp_path)
    prediction = model.predict(_entity(), DOC)

    assert isinstance(prediction, ConfidencePrediction)
    assert prediction.entity_label == "INVOICE_ID"
    assert prediction.entity_text == "INV-2026-001"
    assert 0.0 <= prediction.confidence <= 1.0
    assert prediction.decision in {"auto", "review", "reject"}
    assert prediction.model_version.startswith("rf_confidence_v")


def test_predict_without_model_raises() -> None:
    """predict() before train/load must raise RFModelNotLoadedError."""
    model = RFConfidenceModel(use_bert=False)
    with pytest.raises(RFModelNotLoadedError):
        model.predict(_entity(), DOC)


def test_predict_batch_returns_list(tmp_path: Path) -> None:
    """predict_batch() must return one prediction per entity."""
    model = _trained_model(tmp_path)
    entities = [
        _entity("INVOICE_ID"),
        _entity("INVOICE_DATE", text="2026-01-15", start=27, end=37),
    ]
    predictions = model.predict_batch(entities, DOC)

    assert len(predictions) == 2
    labels = {p.entity_label for p in predictions}
    assert labels == {"INVOICE_ID", "INVOICE_DATE"}


def test_predict_batch_empty_raises(tmp_path: Path) -> None:
    """predict_batch() with an empty list must raise ValueError."""
    model = _trained_model(tmp_path)
    with pytest.raises(ValueError, match="empty"):
        model.predict_batch([], DOC)


# ──────────────────────────────────────────────────────────────────────────────
# Decision routing (architecture decision #8)
# ──────────────────────────────────────────────────────────────────────────────


def test_route_decision_auto(tmp_path: Path) -> None:
    """High-confidence entities must be routed to 'auto'."""
    model = _trained_model(tmp_path)
    # Directly test the routing method with a certainty=1.0 score.
    decision = model._route_decision("INVOICE_ID", 1.0)
    assert decision == "auto"


def test_route_decision_reject(tmp_path: Path) -> None:
    """Very-low-confidence entities must be routed to 'reject'."""
    model = _trained_model(tmp_path)
    decision = model._route_decision("INVOICE_ID", 0.0)
    assert decision == "reject"


def test_route_decision_review(tmp_path: Path) -> None:
    """Mid-confidence entities must be routed to 'review'."""
    model = _trained_model(tmp_path)
    # Default review_min = 0.70, auto = 0.90 — use 0.80 as the mid point.
    decision = model._route_decision("INVOICE_ID", 0.80)
    assert decision == "review"


# ──────────────────────────────────────────────────────────────────────────────
# Load / save round-trip
# ──────────────────────────────────────────────────────────────────────────────


def test_load_save_roundtrip(tmp_path: Path) -> None:
    """A saved model must load and produce the same prediction decision."""
    model_a = RFConfidenceModel(use_bert=False)
    result = model_a.train(_minimal_records(), output_dir=tmp_path, n_estimators=5)
    pred_a = model_a.predict(_entity(), DOC)

    model_b = RFConfidenceModel(use_bert=False)
    model_b.load(Path(result.model_path))
    pred_b = model_b.predict(_entity(), DOC)

    assert pred_a.decision == pred_b.decision
    assert pred_a.confidence == pytest.approx(pred_b.confidence, abs=1e-6)


def test_load_missing_file_raises(tmp_path: Path) -> None:
    """load() must raise FileNotFoundError for a non-existent path."""
    model = RFConfidenceModel(use_bert=False)
    with pytest.raises(FileNotFoundError):
        model.load(tmp_path / "nonexistent.joblib")


# ──────────────────────────────────────────────────────────────────────────────
# Version tag
# ──────────────────────────────────────────────────────────────────────────────


def test_version_tag_format() -> None:
    """_make_version_tag must return a string matching rf_confidence_vYYYYMMDD_HHMM."""
    import re

    tag = RFConfidenceModel._make_version_tag()
    assert re.fullmatch(r"rf_confidence_v\d{8}_\d{4}", tag), f"Unexpected tag: {tag}"


# ──────────────────────────────────────────────────────────────────────────────
# CLI helper: _load_rf_training_records
# ──────────────────────────────────────────────────────────────────────────────


def test_load_rf_training_records_from_sample_fixture() -> None:
    """_load_rf_training_records must parse the sample JSONL fixture."""
    from app.ml.cli import _load_rf_training_records

    fixture = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "annotation"
        / "rf_training_records.jsonl"
    )
    records = _load_rf_training_records(fixture)

    assert len(records) >= 2
    assert all(isinstance(r, TrainingRecord) for r in records)
    labels = {r.entity.label for r in records}
    assert labels >= {"INVOICE_ID", "INVOICE_DATE"}


def test_load_rf_training_records_missing_file(tmp_path: Path) -> None:
    """_load_rf_training_records must raise FileNotFoundError for missing file."""
    from app.ml.cli import _load_rf_training_records

    with pytest.raises(FileNotFoundError):
        _load_rf_training_records(tmp_path / "missing.jsonl")


def test_load_rf_training_records_empty_file_raises(tmp_path: Path) -> None:
    """_load_rf_training_records must raise ValueError for a file with no entities."""
    empty_file = tmp_path / "empty.jsonl"
    empty_file.write_text(
        json.dumps({"text": "hello", "entities": [], "is_correct": True}) + "\n",
        encoding="utf-8",
    )
    from app.ml.cli import _load_rf_training_records

    with pytest.raises(ValueError, match="No training records"):
        _load_rf_training_records(empty_file)


def test_load_rf_training_records_malformed_json_raises(tmp_path: Path) -> None:
    """_load_rf_training_records must raise ValueError for malformed JSON lines."""
    bad_file = tmp_path / "bad.jsonl"
    bad_file.write_text("not-valid-json\n", encoding="utf-8")

    from app.ml.cli import _load_rf_training_records

    with pytest.raises(ValueError, match="Malformed JSON"):
        _load_rf_training_records(bad_file)
