"""Tests for NERKPIService and ExtractionKPIService (Month 3)."""

import pytest

from app.kpi.metrics import (
    EntitySpan,
    ExtractionKPIService,
    NERKPIService,
)


# ──────────────────────────────────────────────────────────────────────────────
# NERKPIService
# ──────────────────────────────────────────────────────────────────────────────

GOLD = [
    EntitySpan("doc-1", "INVOICE_ID", "INV-001"),
    EntitySpan("doc-1", "INVOICE_DATE", "2026-01-15"),
    EntitySpan("doc-1", "TOTAL_AMOUNT", "$1,200.00"),
    EntitySpan("doc-2", "INVOICE_ID", "INV-002"),
    EntitySpan("doc-2", "VENDOR_NAME", "Acme SARL"),
]

PERFECT_PRED = list(GOLD)  # identical spans → P=R=F1=1.0

PARTIAL_PRED = [
    EntitySpan("doc-1", "INVOICE_ID", "INV-001"),   # TP
    EntitySpan("doc-1", "INVOICE_DATE", "2026-01-15"),  # TP
    # TOTAL_AMOUNT missing → FN
    EntitySpan("doc-2", "INVOICE_ID", "INV-002"),   # TP
    EntitySpan("doc-2", "VENDOR_NAME", "Wrong Co"), # FP (wrong value)
]


def test_ner_kpi_perfect_prediction() -> None:
    """Perfect predictions must yield P=R=F1=1.0."""
    report = NERKPIService().evaluate(GOLD, PERFECT_PRED)
    assert report.overall_precision == pytest.approx(1.0)
    assert report.overall_recall == pytest.approx(1.0)
    assert report.overall_f1 == pytest.approx(1.0)


def test_ner_kpi_meets_target_with_perfect_prediction() -> None:
    """meets_target must be True when F1 exceeds the threshold."""
    report = NERKPIService(f1_target=0.85).evaluate(GOLD, PERFECT_PRED)
    assert report.meets_target is True


def test_ner_kpi_partial_prediction_reduces_recall() -> None:
    """Missing a gold span must reduce recall below 1.0."""
    report = NERKPIService().evaluate(GOLD, PARTIAL_PRED)
    assert report.overall_recall < 1.0


def test_ner_kpi_partial_prediction_fp_reduces_precision() -> None:
    """A false positive must reduce precision below 1.0."""
    report = NERKPIService().evaluate(GOLD, PARTIAL_PRED)
    assert report.overall_precision < 1.0


def test_ner_kpi_per_label_contains_all_labels() -> None:
    """per_label must include all labels from gold and predicted."""
    report = NERKPIService().evaluate(GOLD, PARTIAL_PRED)
    label_names = {m.label for m in report.per_label}
    assert {"INVOICE_ID", "INVOICE_DATE", "TOTAL_AMOUNT", "VENDOR_NAME"} <= label_names


def test_ner_kpi_document_count() -> None:
    """document_count must equal the number of unique document IDs."""
    report = NERKPIService().evaluate(GOLD, PERFECT_PRED)
    assert report.document_count == 2


def test_ner_kpi_empty_prediction_zero_scores() -> None:
    """No predictions at all must yield precision=0, recall=0, F1=0."""
    report = NERKPIService().evaluate(GOLD, [])
    assert report.overall_precision == pytest.approx(0.0)
    assert report.overall_recall == pytest.approx(0.0)
    assert report.overall_f1 == pytest.approx(0.0)
    assert report.meets_target is False


def test_ner_kpi_no_gold_and_no_predicted_gives_zero_f1() -> None:
    """Empty gold and predicted must produce a zero-score report safely."""
    report = NERKPIService().evaluate([], [])
    assert report.overall_f1 == pytest.approx(0.0)
    assert report.document_count == 0


def test_ner_kpi_matching_is_case_insensitive() -> None:
    """Value matching must be case-insensitive and strip whitespace."""
    gold = [EntitySpan("doc-1", "INVOICE_ID", "INV-001")]
    pred = [EntitySpan("doc-1", "INVOICE_ID", "  inv-001  ")]
    report = NERKPIService().evaluate(gold, pred)
    assert report.overall_f1 == pytest.approx(1.0)


def test_ner_kpi_does_not_meet_target_below_threshold() -> None:
    """meets_target must be False when F1 is below f1_target."""
    report = NERKPIService(f1_target=0.99).evaluate(GOLD, PARTIAL_PRED)
    assert report.meets_target is False


def test_ner_kpi_from_field_decisions() -> None:
    """from_field_decisions must evaluate a single document correctly."""
    from app.pipeline.decision_engine import FieldDecision

    gold_fields = {"INVOICE_ID": "INV-001", "INVOICE_DATE": "2026-01-15"}
    pipeline_fields = [
        FieldDecision(
            field_name="INVOICE_ID",
            value="INV-001",
            confidence=0.95,
            decision="auto",
            scorer="heuristic",
        ),
        FieldDecision(
            field_name="INVOICE_DATE",
            value="2026-01-15",
            confidence=0.88,
            decision="auto",
            scorer="heuristic",
        ),
    ]
    report = NERKPIService().from_field_decisions("doc-1", gold_fields, pipeline_fields)
    assert report.overall_f1 == pytest.approx(1.0)


# ──────────────────────────────────────────────────────────────────────────────
# ExtractionKPIService
# ──────────────────────────────────────────────────────────────────────────────


def test_extraction_kpi_computes_rates_correctly() -> None:
    """Auto/review/reject rates must sum to ~1.0 and be individually correct."""
    kpi = ExtractionKPIService().from_aggregates(
        decision_counts={"auto": 80, "review": 15, "reject": 5},
    )
    assert kpi.total_documents == 100
    assert kpi.auto_rate == pytest.approx(0.80, abs=1e-4)
    assert kpi.review_rate == pytest.approx(0.15, abs=1e-4)
    assert kpi.reject_rate == pytest.approx(0.05, abs=1e-4)
    assert pytest.approx(kpi.auto_rate + kpi.review_rate + kpi.reject_rate, abs=1e-4) == 1.0


def test_extraction_kpi_zero_documents() -> None:
    """All rates must be 0.0 when no documents exist."""
    kpi = ExtractionKPIService().from_aggregates(decision_counts={})
    assert kpi.total_documents == 0
    assert kpi.auto_rate == pytest.approx(0.0)
    assert kpi.review_rate == pytest.approx(0.0)
    assert kpi.reject_rate == pytest.approx(0.0)


def test_extraction_kpi_includes_confidence_by_field() -> None:
    """avg confidence per field must be preserved in the report."""
    avg_conf = {"INVOICE_ID": 0.93, "TOTAL_AMOUNT": 0.87}
    kpi = ExtractionKPIService().from_aggregates(
        decision_counts={"auto": 10},
        avg_confidence_by_field=avg_conf,
    )
    assert kpi.average_confidence_by_field["INVOICE_ID"] == pytest.approx(0.93)
    assert kpi.average_confidence_by_field["TOTAL_AMOUNT"] == pytest.approx(0.87)


def test_extraction_kpi_scorer_distribution_preserved() -> None:
    """scorer_distribution must be passed through unchanged."""
    kpi = ExtractionKPIService().from_aggregates(
        decision_counts={"auto": 60, "review": 40},
        scorer_distribution={"rf": 80, "heuristic": 20},
    )
    assert kpi.scorer_distribution == {"rf": 80, "heuristic": 20}
