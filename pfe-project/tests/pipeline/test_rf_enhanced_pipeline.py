"""Tests for the RF-enhanced sequential extraction decision engine.

These tests exercise the new ``rf_model`` code path in
``SequentialExtractionDecisionEngine``.  All tests use:

* A ``_FakeExtractor`` that returns controlled entities (no I/O, no spaCy).
* A real but tiny ``RFConfidenceModel`` trained in-test (5 trees, no BERT)
  so that the RF prediction path is truly exercised without mocking sklearn.
* ``use_bert=False`` everywhere for speed.

The existing heuristic-path tests in ``test_decision_engine.py`` are NOT
duplicated here — they continue to cover the fallback path.
"""

from pathlib import Path
from typing import List

import pytest

from app.ml.ner_extractor import ExtractedEntity, ExtractionResult
from app.ml.rf_confidence_model import RFConfidenceModel, TrainingRecord
from app.pipeline.decision_engine import (
    FieldDecision,
    RFModelLoader,
    SequentialExtractionDecisionEngine,
)


# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

DOC = "Invoice INV-2026-001 dated 2026-01-15. Vendor: Acme SARL. Total: $1,200.00"


class _FakeExtractor:
    """Minimal extractor stub — returns a fixed entity list."""

    def __init__(self, entities: List[ExtractedEntity]) -> None:
        self._entities = entities

    def extract(self, text: str) -> ExtractionResult:
        return ExtractionResult(text=text, entities=self._entities)


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


def _build_trained_rf(tmp_path: Path) -> RFConfidenceModel:
    """Train a minimal RF model (5 trees, no BERT) and return it loaded."""
    records = [
        TrainingRecord(entity=_entity("INVOICE_ID"), document_text=DOC, is_correct=True),
        TrainingRecord(entity=_entity("INVOICE_DATE", "2026-01-15", 27, 37), document_text=DOC, is_correct=True),
        TrainingRecord(entity=_entity("TOTAL_AMOUNT", "$1,200.00", 64, 73), document_text=DOC, is_correct=True),
        TrainingRecord(entity=_entity("VENDOR_NAME", "Acme SARL", 47, 56), document_text=DOC, is_correct=True),
        TrainingRecord(entity=_entity("INVOICE_ID", "?", 0, 1, score=0.1), document_text=DOC, is_correct=False),
        TrainingRecord(entity=_entity("TOTAL_AMOUNT", "x", 0, 1, score=0.1), document_text=DOC, is_correct=False),
    ]
    rf = RFConfidenceModel(use_bert=False)
    rf.train(records, output_dir=tmp_path, n_estimators=5)
    return rf


# ──────────────────────────────────────────────────────────────────────────────
# RF engine initialisation
# ──────────────────────────────────────────────────────────────────────────────


def test_engine_reports_rf_scorer_when_model_injected(tmp_path: Path) -> None:
    """Engine._active_scorer must be 'rf' when rf_model is provided."""
    rf = _build_trained_rf(tmp_path)
    engine = SequentialExtractionDecisionEngine(
        extractor=_FakeExtractor([_entity()]),
        rf_model=rf,
    )
    assert engine._active_scorer == "rf"


def test_engine_reports_heuristic_scorer_by_default() -> None:
    """Engine._active_scorer must default to 'heuristic' without rf_model."""
    engine = SequentialExtractionDecisionEngine(extractor=_FakeExtractor([]))
    assert engine._active_scorer == "heuristic"


# ──────────────────────────────────────────────────────────────────────────────
# RF scoring path — run() output structure
# ──────────────────────────────────────────────────────────────────────────────


def test_rf_engine_run_returns_pipeline_decision_result(tmp_path: Path) -> None:
    """run() with RF model must return a PipelineDecisionResult."""
    rf = _build_trained_rf(tmp_path)
    engine = SequentialExtractionDecisionEngine(
        extractor=_FakeExtractor([_entity()]),
        rf_model=rf,
    )
    result = engine.run(DOC)

    assert result.text == DOC
    assert len(result.fields) == 1
    assert result.scorer == "rf"
    assert result.overall_decision in {"auto", "review", "reject"}


def test_rf_engine_field_decision_has_rf_scorer_tag(tmp_path: Path) -> None:
    """Every FieldDecision produced in RF mode must carry scorer='rf'."""
    rf = _build_trained_rf(tmp_path)
    entities = [_entity("INVOICE_ID"), _entity("INVOICE_DATE", "2026-01-15", 27, 37)]
    engine = SequentialExtractionDecisionEngine(
        extractor=_FakeExtractor(entities),
        rf_model=rf,
    )
    result = engine.run(DOC)

    for fd in result.fields:
        assert fd.scorer == "rf", f"Expected scorer='rf' for {fd.field_name}, got {fd.scorer!r}"


def test_rf_engine_confidence_is_in_valid_range(tmp_path: Path) -> None:
    """RF-predicted confidence must be in [0.0, 1.0] for every field."""
    rf = _build_trained_rf(tmp_path)
    entities = [
        _entity("INVOICE_ID"),
        _entity("TOTAL_AMOUNT", "$1,200.00", 64, 73),
    ]
    engine = SequentialExtractionDecisionEngine(
        extractor=_FakeExtractor(entities),
        rf_model=rf,
    )
    result = engine.run(DOC)

    for fd in result.fields:
        assert 0.0 <= fd.confidence <= 1.0, (
            f"{fd.field_name} confidence {fd.confidence} outside [0, 1]"
        )


def test_rf_engine_confidence_factors_contains_rf_key(tmp_path: Path) -> None:
    """confidence_factors for RF-scored fields must contain 'rf_confidence'."""
    rf = _build_trained_rf(tmp_path)
    engine = SequentialExtractionDecisionEngine(
        extractor=_FakeExtractor([_entity()]),
        rf_model=rf,
    )
    result = engine.run(DOC)

    assert "rf_confidence" in result.fields[0].confidence_factors


def test_rf_engine_decision_matches_confidence_and_thresholds(tmp_path: Path) -> None:
    """RF routing must respect the configured threshold bands."""
    rf = _build_trained_rf(tmp_path)
    engine = SequentialExtractionDecisionEngine(
        extractor=_FakeExtractor([_entity()]),
        rf_model=rf,
    )
    result = engine.run(DOC)
    fd = result.fields[0]

    # The decision must be consistent with the reported confidence value.
    thresholds = engine._resolve_thresholds(fd.field_name)
    if fd.confidence >= thresholds["auto"]:
        assert fd.decision == "auto"
    elif fd.confidence >= thresholds["review_min"]:
        assert fd.decision == "review"
    else:
        assert fd.decision == "reject"


# ──────────────────────────────────────────────────────────────────────────────
# Hot-swap and rollback
# ──────────────────────────────────────────────────────────────────────────────


def test_switch_to_rf_model_changes_active_scorer(tmp_path: Path) -> None:
    """switch_to_rf_model() must flip _active_scorer to 'rf'."""
    rf = _build_trained_rf(tmp_path)
    engine = SequentialExtractionDecisionEngine(extractor=_FakeExtractor([_entity()]))
    assert engine._active_scorer == "heuristic"

    engine.switch_to_rf_model(rf)
    assert engine._active_scorer == "rf"
    assert engine.rf_model is rf


def test_switch_to_heuristic_rolls_back_scorer(tmp_path: Path) -> None:
    """switch_to_heuristic() must revert _active_scorer to 'heuristic'."""
    rf = _build_trained_rf(tmp_path)
    engine = SequentialExtractionDecisionEngine(
        extractor=_FakeExtractor([_entity()]),
        rf_model=rf,
    )
    assert engine._active_scorer == "rf"

    engine.switch_to_heuristic()
    assert engine._active_scorer == "heuristic"
    assert engine.rf_model is None


def test_engine_after_rollback_uses_heuristic_path(tmp_path: Path) -> None:
    """After switch_to_heuristic(), run() must produce scorer='heuristic' output."""
    rf = _build_trained_rf(tmp_path)
    engine = SequentialExtractionDecisionEngine(
        extractor=_FakeExtractor([_entity()]),
        rf_model=rf,
    )
    engine.switch_to_heuristic()
    result = engine.run(DOC)

    assert result.scorer == "heuristic"
    assert result.fields[0].scorer == "heuristic"


# ──────────────────────────────────────────────────────────────────────────────
# RFModelLoader
# ──────────────────────────────────────────────────────────────────────────────


def test_rf_model_loader_returns_none_when_no_model_exists(tmp_path: Path) -> None:
    """RFModelLoader.load_latest() must return None when no model is saved."""
    from app.config import Settings
    settings = Settings(RF_MODEL_OUTPUT_DIR=str(tmp_path))
    loader = RFModelLoader(settings=settings, use_bert=False)

    result = loader.load_latest()
    assert result is None


def test_rf_model_loader_loads_trained_model(tmp_path: Path) -> None:
    """RFModelLoader.load_latest() must load and return a trained model."""
    from app.config import Settings

    # Train and save a model into tmp_path.
    rf = _build_trained_rf(tmp_path)

    settings = Settings(RF_MODEL_OUTPUT_DIR=str(tmp_path))
    loader = RFModelLoader(settings=settings, use_bert=False)

    loaded = loader.load_latest()
    assert loaded is not None
    # Verify the loaded model can predict.
    prediction = loaded.predict(_entity(), DOC)
    assert prediction.entity_label == "INVOICE_ID"


def test_rf_model_loader_lists_available_models(tmp_path: Path) -> None:
    """RFModelLoader.list_available() must return filenames of saved models."""
    from app.config import Settings

    _build_trained_rf(tmp_path)  # creates one .joblib file

    settings = Settings(RF_MODEL_OUTPUT_DIR=str(tmp_path))
    loader = RFModelLoader(settings=settings, use_bert=False)

    available = loader.list_available()
    assert len(available) >= 1
    assert all(name.startswith("rf_confidence_v") for name in available)
    assert all(name.endswith(".joblib") for name in available)


def test_rf_model_loader_empty_dir_returns_empty_list(tmp_path: Path) -> None:
    """list_available() must return [] when the directory exists but is empty."""
    from app.config import Settings
    settings = Settings(RF_MODEL_OUTPUT_DIR=str(tmp_path))
    loader = RFModelLoader(settings=settings, use_bert=False)

    assert loader.list_available() == []


# ──────────────────────────────────────────────────────────────────────────────
# Fallback path: RF model raises RFModelNotLoadedError mid-run
# ──────────────────────────────────────────────────────────────────────────────


def test_engine_falls_back_to_heuristic_if_rf_not_loaded(tmp_path: Path) -> None:
    """Engine must gracefully fall back when RF model is unloaded during run."""
    # Inject an unloaded (bare) RF model — predict() will raise RFModelNotLoadedError.
    unloaded_rf = RFConfidenceModel(use_bert=False)  # no train/load called
    engine = SequentialExtractionDecisionEngine(
        extractor=_FakeExtractor([_entity()]),
        rf_model=unloaded_rf,
    )

    # Must not raise — falls back to heuristic per-field.
    result = engine.run(DOC)
    assert result.fields[0].scorer == "heuristic"
    assert result.fields[0].decision in {"auto", "review", "reject"}
