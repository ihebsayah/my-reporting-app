"""Tests for the sequential extraction decision engine."""

from typing import List

from app.config import Settings
from app.ml.ner_extractor import ExtractedEntity, ExtractionResult
from app.pipeline.decision_engine import SequentialExtractionDecisionEngine


class _FakeExtractor:
    """Minimal extractor stub for pipeline tests."""

    def __init__(self, entities: List[ExtractedEntity]) -> None:
        self._result = ExtractionResult(text="sample", entities=entities)

    def extract(self, text: str) -> ExtractionResult:
        return ExtractionResult(text=text, entities=self._result.entities)


def test_pipeline_assigns_auto_review_and_reject_decisions() -> None:
    """Ensure field decisions follow the locked threshold bands."""
    entities = [
        ExtractedEntity(0, 5, "INV-1", "INVOICE_ID", ("regex", "spacy"), 0.95),
        ExtractedEntity(10, 20, "2026-03-29", "INVOICE_DATE", ("regex",), 0.85),
        ExtractedEntity(25, 31, "$1.00", "TOTAL_AMOUNT", ("spacy",), 0.65),
    ]
    settings = Settings()
    engine = SequentialExtractionDecisionEngine(
        settings=settings,
        extractor=_FakeExtractor(entities),
    )

    result = engine.run("Invoice text")
    decisions = {field.field_name: field.decision for field in result.fields}

    assert decisions["INVOICE_ID"] == "auto"
    assert decisions["INVOICE_DATE"] == "review"
    assert decisions["TOTAL_AMOUNT"] == "reject"
    assert result.overall_decision == "reject"


def test_pipeline_selects_best_entity_per_field() -> None:
    """Ensure the highest-confidence entity wins per field."""
    entities = [
        ExtractedEntity(0, 5, "INV-LOW", "INVOICE_ID", ("regex",), 0.80),
        ExtractedEntity(6, 14, "INV-HIGH", "INVOICE_ID", ("regex", "spacy"), 0.95),
    ]
    engine = SequentialExtractionDecisionEngine(extractor=_FakeExtractor(entities))

    result = engine.run("Invoice text")

    assert len(result.fields) == 1
    assert result.fields[0].value == "INV-HIGH"
    assert result.fields[0].decision == "auto"


def test_pipeline_returns_reject_when_no_entities_found() -> None:
    """Ensure empty extraction results default to reject."""
    engine = SequentialExtractionDecisionEngine(extractor=_FakeExtractor([]))

    result = engine.run("No invoice fields here")

    assert result.fields == []
    assert result.overall_decision == "reject"


def test_pipeline_uses_per_field_thresholds() -> None:
    """Ensure per-field thresholds override the global defaults."""
    entities = [
        ExtractedEntity(0, 10, "Acme Corp", "VENDOR_NAME", ("regex",), 0.89),
        ExtractedEntity(11, 17, "$99.00", "TOTAL_AMOUNT", ("regex",), 0.89),
    ]
    settings = Settings(
        FIELD_THRESHOLDS_JSON=(
            '{"VENDOR_NAME":{"auto":0.88,"review_min":0.68,"review_max":0.87},'
            '"TOTAL_AMOUNT":{"auto":0.95,"review_min":0.80,"review_max":0.94}}'
        )
    )
    engine = SequentialExtractionDecisionEngine(
        settings=settings,
        extractor=_FakeExtractor(entities),
    )

    result = engine.run("Invoice text")
    decisions = {field.field_name: field.decision for field in result.fields}

    assert decisions["VENDOR_NAME"] == "auto"
    assert decisions["TOTAL_AMOUNT"] == "review"
    assert result.overall_decision == "review"
