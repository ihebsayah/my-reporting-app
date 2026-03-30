"""Sequential extraction decision engine."""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from app.config import Settings, get_settings
from app.ml.confidence_scorer import FieldConfidenceScorer
from app.ml.ner_extractor import ExtractedEntity, ExtractionResult, RegexSpacyEnsembleExtractor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FieldDecision:
    """Represents the decision for a single extracted field."""

    field_name: str
    value: Optional[str]
    confidence: float
    decision: str
    sources: List[str] = field(default_factory=list)
    confidence_factors: Dict[str, float] = field(default_factory=dict)
    start: Optional[int] = None
    end: Optional[int] = None


@dataclass(frozen=True)
class PipelineDecisionResult:
    """Represents the full sequential pipeline output for one document."""

    text: str
    fields: List[FieldDecision]
    overall_decision: str


class SequentialExtractionDecisionEngine:
    """Apply threshold-based routing on top of ensemble NER extraction."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        extractor: Optional[RegexSpacyEnsembleExtractor] = None,
        confidence_scorer: Optional[FieldConfidenceScorer] = None,
    ) -> None:
        """Initialize the decision engine.

        Args:
            settings: Optional application settings override.
            extractor: Optional ensemble extractor override.
            confidence_scorer: Optional confidence scorer override.
        """
        self.settings = settings or get_settings()
        self.extractor = extractor or RegexSpacyEnsembleExtractor(settings=self.settings)
        self.confidence_scorer = confidence_scorer or FieldConfidenceScorer(
            settings=self.settings
        )
        self._field_thresholds = self.settings.field_thresholds()

    def run(self, text: str) -> PipelineDecisionResult:
        """Run the sequential extraction and decision flow.

        Args:
            text: Raw document text.

        Returns:
            Pipeline output with field-level and overall decisions.
        """
        extraction_result = self.extractor.extract(text)
        field_decisions = self._build_field_decisions(extraction_result)
        overall_decision = self._aggregate_overall_decision(field_decisions)
        logger.info(
            "Pipeline completed with overall decision '%s' across %d fields.",
            overall_decision,
            len(field_decisions),
        )
        return PipelineDecisionResult(
            text=text,
            fields=field_decisions,
            overall_decision=overall_decision,
        )

    def _build_field_decisions(
        self, extraction_result: ExtractionResult
    ) -> List[FieldDecision]:
        """Select the best entity per field and map it to a threshold decision."""
        grouped: Dict[str, List[ExtractedEntity]] = {}
        for entity in extraction_result.entities:
            grouped.setdefault(entity.label, []).append(entity)

        decisions: List[FieldDecision] = []
        for field_name in sorted(grouped):
            best_entity = max(
                grouped[field_name],
                key=lambda item: (item.score, len(item.sources), -item.start),
            )
            decisions.append(self._entity_to_decision(field_name, best_entity))
        return decisions

    def _entity_to_decision(
        self, field_name: str, entity: ExtractedEntity
    ) -> FieldDecision:
        """Convert one extracted entity into a field decision."""
        assessment = self.confidence_scorer.score_entity(entity)
        confidence = assessment.confidence
        thresholds = self._resolve_thresholds(field_name)
        if confidence >= thresholds["auto"]:
            decision = "auto"
        elif thresholds["review_min"] <= confidence <= thresholds["review_max"]:
            decision = "review"
        else:
            decision = "reject"

        return FieldDecision(
            field_name=field_name,
            value=entity.text,
            confidence=confidence,
            decision=decision,
            sources=list(entity.sources),
            confidence_factors=assessment.factors,
            start=entity.start,
            end=entity.end,
        )

    def _resolve_thresholds(self, field_name: str) -> Dict[str, float]:
        """Return thresholds for a field with fallback to global defaults."""
        return self._field_thresholds.get(
            field_name,
            {
                "auto": self.settings.auto_approval_threshold,
                "review_min": self.settings.review_min_threshold,
                "review_max": self.settings.review_max_threshold,
            },
        )

    @staticmethod
    def _aggregate_overall_decision(field_decisions: Sequence[FieldDecision]) -> str:
        """Aggregate field-level decisions into a document-level outcome."""
        decision_priority = {"reject": 0, "review": 1, "auto": 2}
        if not field_decisions:
            return "reject"
        return min(field_decisions, key=lambda item: decision_priority[item.decision]).decision
