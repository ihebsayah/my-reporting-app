"""Tests for field confidence scoring."""

from app.ml.confidence_scorer import FieldConfidenceScorer
from app.ml.ner_extractor import ExtractedEntity


def test_confidence_scorer_rewards_multi_source_entities() -> None:
    """Ensure multi-source entities receive a higher confidence."""
    scorer = FieldConfidenceScorer()
    regex_only = ExtractedEntity(0, 5, "INV-1", "INVOICE_ID", ("regex",), 0.85)
    merged = ExtractedEntity(0, 5, "INV-1", "INVOICE_ID", ("regex", "spacy"), 0.85)

    regex_assessment = scorer.score_entity(regex_only)
    merged_assessment = scorer.score_entity(merged)

    assert merged_assessment.confidence > regex_assessment.confidence
    assert merged_assessment.factors["multi_source_bonus"] > 0


def test_confidence_scorer_penalizes_very_short_values() -> None:
    """Ensure implausibly short values get a penalty."""
    scorer = FieldConfidenceScorer()
    entity = ExtractedEntity(0, 1, "A", "VENDOR_NAME", ("regex",), 0.8)

    assessment = scorer.score_entity(entity)

    assert assessment.factors["length_penalty"] < 0
    assert assessment.confidence < 0.83
