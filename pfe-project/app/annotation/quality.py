"""Annotation quality metrics and agreement helpers."""

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FieldPresenceComparison:
    """Represents presence labels for one field across two annotators."""

    document_id: str
    field_name: str
    annotator_a_present: int
    annotator_b_present: int


@dataclass(frozen=True)
class AgreementResult:
    """Represents agreement metrics for a single field."""

    field_name: str
    observed_agreement: float
    expected_agreement: float
    cohen_kappa: float
    comparisons: int


def calculate_cohen_kappa(labels_a: Sequence[int], labels_b: Sequence[int]) -> float:
    """Calculate Cohen's Kappa for binary labels.

    Args:
        labels_a: Binary labels from annotator A.
        labels_b: Binary labels from annotator B.

    Returns:
        Cohen's Kappa score.

    Raises:
        ValueError: If the label sequences are empty, mismatched, or non-binary.
    """
    _validate_binary_labels(labels_a, labels_b)
    total = len(labels_a)
    agreement = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / total

    prob_a_positive = sum(labels_a) / total
    prob_b_positive = sum(labels_b) / total
    prob_a_negative = 1.0 - prob_a_positive
    prob_b_negative = 1.0 - prob_b_positive
    expected = (prob_a_positive * prob_b_positive) + (
        prob_a_negative * prob_b_negative
    )

    if expected == 1.0:
        logger.warning("Expected agreement is 1.0; returning perfect agreement fallback.")
        return 1.0

    kappa = (agreement - expected) / (1.0 - expected)
    logger.debug(
        "Calculated Cohen's Kappa with observed=%s expected=%s kappa=%s.",
        agreement,
        expected,
        kappa,
    )
    return kappa


def summarize_field_agreement(
    comparisons: Iterable[FieldPresenceComparison],
) -> Dict[str, AgreementResult]:
    """Aggregate binary field-presence comparisons into field-level agreement metrics.

    Args:
        comparisons: Field-level comparisons across annotators.

    Returns:
        Agreement metrics keyed by field name.

    Raises:
        ValueError: If no comparisons are provided for a field group.
    """
    grouped: Dict[str, List[FieldPresenceComparison]] = {}
    for comparison in comparisons:
        grouped.setdefault(comparison.field_name, []).append(comparison)

    results: Dict[str, AgreementResult] = {}
    for field_name, field_comparisons in grouped.items():
        if not field_comparisons:
            logger.error("No comparisons found for field '%s'.", field_name)
            raise ValueError(f"No comparisons found for field '{field_name}'.")

        labels_a = [item.annotator_a_present for item in field_comparisons]
        labels_b = [item.annotator_b_present for item in field_comparisons]
        kappa = calculate_cohen_kappa(labels_a, labels_b)
        observed, expected = _agreement_components(labels_a, labels_b)
        results[field_name] = AgreementResult(
            field_name=field_name,
            observed_agreement=observed,
            expected_agreement=expected,
            cohen_kappa=kappa,
            comparisons=len(field_comparisons),
        )
        logger.info(
            "Field '%s' agreement computed with kappa=%.3f across %d comparisons.",
            field_name,
            kappa,
            len(field_comparisons),
        )

    return results


def build_presence_comparisons(
    annotator_a: Dict[str, Sequence[str]],
    annotator_b: Dict[str, Sequence[str]],
) -> List[FieldPresenceComparison]:
    """Convert per-document field lists into field presence comparisons.

    Args:
        annotator_a: Mapping of document IDs to annotated field labels for annotator A.
        annotator_b: Mapping of document IDs to annotated field labels for annotator B.

    Returns:
        Normalized field-presence comparisons ready for agreement metrics.
    """
    document_ids = sorted(set(annotator_a) | set(annotator_b))
    all_fields = sorted(
        set(_flatten_field_values(annotator_a.values()))
        | set(_flatten_field_values(annotator_b.values()))
    )
    comparisons: List[FieldPresenceComparison] = []

    for document_id in document_ids:
        fields_a = set(annotator_a.get(document_id, []))
        fields_b = set(annotator_b.get(document_id, []))
        for field_name in all_fields:
            comparisons.append(
                FieldPresenceComparison(
                    document_id=document_id,
                    field_name=field_name,
                    annotator_a_present=int(field_name in fields_a),
                    annotator_b_present=int(field_name in fields_b),
                )
            )
    logger.info(
        "Built %d field presence comparisons for %d documents.",
        len(comparisons),
        len(document_ids),
    )
    return comparisons


def _flatten_field_values(values: Iterable[Sequence[str]]) -> List[str]:
    """Flatten nested field label sequences into a single list."""
    flattened: List[str] = []
    for sequence in values:
        flattened.extend(sequence)
    return flattened


def _validate_binary_labels(labels_a: Sequence[int], labels_b: Sequence[int]) -> None:
    """Validate binary label arrays before kappa calculation."""
    if not labels_a or not labels_b:
        logger.error("Cannot calculate Cohen's Kappa on empty labels.")
        raise ValueError("Label sequences must not be empty.")
    if len(labels_a) != len(labels_b):
        logger.error("Mismatched label lengths: %d != %d", len(labels_a), len(labels_b))
        raise ValueError("Label sequences must have the same length.")
    valid_values = {0, 1}
    if set(labels_a) - valid_values or set(labels_b) - valid_values:
        logger.error("Non-binary labels detected in agreement calculation.")
        raise ValueError("Label sequences must contain only binary values 0 and 1.")


def _agreement_components(
    labels_a: Sequence[int], labels_b: Sequence[int]
) -> Tuple[float, float]:
    """Return observed and expected agreement terms used by Cohen's Kappa."""
    total = len(labels_a)
    observed = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / total
    prob_a_positive = sum(labels_a) / total
    prob_b_positive = sum(labels_b) / total
    prob_a_negative = 1.0 - prob_a_positive
    prob_b_negative = 1.0 - prob_b_positive
    expected = (prob_a_positive * prob_b_positive) + (
        prob_a_negative * prob_b_negative
    )
    return observed, expected
