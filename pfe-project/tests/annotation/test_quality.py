"""Tests for annotation agreement metrics."""

import pytest

from app.annotation.quality import (
    build_presence_comparisons,
    calculate_cohen_kappa,
    summarize_field_agreement,
)


def test_calculate_cohen_kappa_returns_expected_value() -> None:
    """Ensure Cohen's Kappa is computed correctly for binary labels."""
    kappa = calculate_cohen_kappa([1, 1, 0, 0], [1, 0, 0, 0])

    assert round(kappa, 3) == 0.5


def test_calculate_cohen_kappa_rejects_invalid_labels() -> None:
    """Ensure invalid label inputs are rejected."""
    with pytest.raises(ValueError):
        calculate_cohen_kappa([1, 2], [1, 0])


def test_build_presence_comparisons_and_summary() -> None:
    """Ensure field presence comparisons aggregate into field-level metrics."""
    annotator_a = {
        "doc-001": ["INVOICE_ID", "TOTAL_AMOUNT"],
        "doc-002": ["VENDOR_NAME"],
    }
    annotator_b = {
        "doc-001": ["INVOICE_ID", "TOTAL_AMOUNT"],
        "doc-002": ["TOTAL_AMOUNT"],
    }

    comparisons = build_presence_comparisons(annotator_a, annotator_b)
    summary = summarize_field_agreement(comparisons)

    assert len(comparisons) == 6
    assert "INVOICE_ID" in summary
    assert summary["TOTAL_AMOUNT"].comparisons == 2
    assert summary["INVOICE_ID"].cohen_kappa == 1.0
