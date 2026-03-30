"""Field-level confidence scoring for extraction results."""

import logging
from dataclasses import dataclass
from typing import Dict, Optional

from app.config import Settings, get_settings
from app.ml.ner_extractor import ExtractedEntity

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConfidenceAssessment:
    """Represents a confidence score and its contributing factors."""

    field_name: str
    confidence: float
    factors: Dict[str, float]


class FieldConfidenceScorer:
    """Compute field-level confidence from extraction signals."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """Initialize the scorer.

        Args:
            settings: Optional application settings override.
        """
        self.settings = settings or get_settings()

    def score_entity(self, entity: ExtractedEntity) -> ConfidenceAssessment:
        """Compute confidence for a single extracted entity.

        Args:
            entity: Extracted entity to score.

        Returns:
            Confidence assessment with factor contributions.
        """
        factors = {
            "base_score": entity.score,
            "multi_source_bonus": 0.08 if len(entity.sources) > 1 else 0.0,
            "regex_pattern_bonus": 0.03 if "regex" in entity.sources else 0.0,
            "length_penalty": self._length_penalty(entity.text),
        }
        confidence = max(0.0, min(0.99, sum(factors.values())))
        logger.debug(
            "Confidence for %s computed as %.3f with factors %s.",
            entity.label,
            confidence,
            factors,
        )
        return ConfidenceAssessment(
            field_name=entity.label,
            confidence=confidence,
            factors=factors,
        )

    @staticmethod
    def _length_penalty(value: str) -> float:
        """Apply a small penalty to implausibly short extracted values."""
        return -0.05 if len(value.strip()) < 3 else 0.0
