"""Sequential extraction decision engine.

Architecture decision #7: pipeline is sequential (NER → confidence → decision).

The engine supports two confidence-scoring modes, selected at construction time:

* **Heuristic mode (default)** — uses ``FieldConfidenceScorer``, the
  lightweight rule-based scorer built in Month 1.  Works with zero trained
  models and is the default when no ``rf_model`` is supplied.

* **RF mode** — uses ``RFConfidenceModel.predict()`` when a pre-loaded
  ``RFConfidenceModel`` is injected.  This is the Month 2+ target path;
  confidence scores and routing decisions come from the trained Random Forest
  rather than hand-tuned heuristics.

Switching between modes is transparent to callers: the ``run()`` signature
and return type are identical in both cases.  All existing tests that do not
supply an ``rf_model`` continue to exercise the heuristic path unchanged.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from app.config import Settings, get_settings
from app.ml.confidence_scorer import FieldConfidenceScorer
from app.ml.ner_extractor import ExtractedEntity, ExtractionResult, RegexSpacyEnsembleExtractor
from app.ml.rf_confidence_model import RFConfidenceModel, RFModelNotLoadedError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FieldDecision:
    """Represents the decision for a single extracted field.

    Attributes:
        field_name: NER label (e.g. ``INVOICE_ID``).
        value: Extracted text value, or ``None`` when no entity was found.
        confidence: Confidence score in [0.0, 1.0].
        decision: One of ``auto``, ``review``, or ``reject``.
        sources: Extraction sources that contributed (e.g. ``["regex", "spacy"]``).
        confidence_factors: Named breakdown of confidence signal contributions.
        start: Char start offset of the entity in the source text.
        end: Char end offset of the entity in the source text.
        scorer: Which scorer produced this confidence (``"heuristic"`` or
            ``"rf"``).  Useful for tracing and monitoring.
    """

    field_name: str
    value: Optional[str]
    confidence: float
    decision: str
    sources: List[str] = field(default_factory=list)
    confidence_factors: Dict[str, float] = field(default_factory=dict)
    start: Optional[int] = None
    end: Optional[int] = None
    scorer: str = "heuristic"


@dataclass(frozen=True)
class PipelineDecisionResult:
    """Represents the full sequential pipeline output for one document.

    Attributes:
        text: The original input text.
        fields: Per-field decisions.
        overall_decision: Document-level routing decision (worst-case field).
        scorer: Which scoring mode was active for this run.
    """

    text: str
    fields: List[FieldDecision]
    overall_decision: str
    scorer: str = "heuristic"


class SequentialExtractionDecisionEngine:
    """Apply threshold-based routing on top of ensemble NER extraction.

    Supports two scoring modes selected by whether ``rf_model`` is supplied:

    * No ``rf_model`` → ``FieldConfidenceScorer`` (heuristic, always available)
    * ``rf_model`` provided → ``RFConfidenceModel.predict()`` (Month 2+ target)

    Args:
        settings: Optional application settings override.
        extractor: Optional ensemble extractor override (useful in tests).
        confidence_scorer: Optional heuristic scorer override.
        rf_model: Optional pre-loaded ``RFConfidenceModel``.  When provided
            the engine switches to RF mode for all confidence scoring.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        extractor: Optional[RegexSpacyEnsembleExtractor] = None,
        confidence_scorer: Optional[FieldConfidenceScorer] = None,
        rf_model: Optional[RFConfidenceModel] = None,
    ) -> None:
        """Initialize the decision engine.

        Args:
            settings: Optional application settings override.
            extractor: Optional ensemble extractor override.
            confidence_scorer: Optional heuristic scorer override.
            rf_model: Optional pre-loaded RF confidence model.
        """
        self.settings = settings or get_settings()
        self.extractor = extractor or RegexSpacyEnsembleExtractor(settings=self.settings)
        self.confidence_scorer = confidence_scorer or FieldConfidenceScorer(
            settings=self.settings
        )
        self.rf_model: Optional[RFConfidenceModel] = rf_model
        self._field_thresholds = self.settings.field_thresholds()
        self._active_scorer = "rf" if rf_model is not None else "heuristic"
        logger.info(
            "SequentialExtractionDecisionEngine initialised with scorer='%s'.",
            self._active_scorer,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def run(self, text: str) -> PipelineDecisionResult:
        """Run the sequential extraction and decision flow.

        Steps (architecture decision #7):
          1. NER extraction (regex + spaCy ensemble)
          2. Confidence scoring (heuristic or RF)
          3. Per-field threshold routing
          4. Document-level decision aggregation

        Args:
            text: Raw document text.

        Returns:
            Pipeline output with field-level and overall decisions.
        """
        extraction_result = self.extractor.extract(text)
        field_decisions = self._build_field_decisions(extraction_result)
        overall_decision = self._aggregate_overall_decision(field_decisions)
        logger.info(
            "Pipeline completed scorer='%s' overall='%s' fields=%d.",
            self._active_scorer,
            overall_decision,
            len(field_decisions),
        )
        return PipelineDecisionResult(
            text=text,
            fields=field_decisions,
            overall_decision=overall_decision,
            scorer=self._active_scorer,
        )

    def switch_to_rf_model(self, rf_model: RFConfidenceModel) -> None:
        """Hot-swap the RF confidence model without recreating the engine.

        This is used by the canary/A-B rollout logic (Month 4) and by
        the monthly retraining pipeline to activate a freshly trained model
        without restarting the server.

        Args:
            rf_model: A pre-loaded ``RFConfidenceModel`` instance.
        """
        self.rf_model = rf_model
        self._active_scorer = "rf"
        logger.info("Engine switched to RF confidence scorer.")

    def switch_to_heuristic(self) -> None:
        """Fall back to the heuristic scorer (safe rollback path).

        Called during monitoring-triggered rollback or canary abort.
        """
        self.rf_model = None
        self._active_scorer = "heuristic"
        logger.warning("Engine rolled back to heuristic confidence scorer.")

    # ──────────────────────────────────────────────────────────────────────────
    # Internal: field decision building
    # ──────────────────────────────────────────────────────────────────────────

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
            if self.rf_model is not None:
                decisions.append(
                    self._entity_to_decision_rf(
                        field_name, best_entity, extraction_result.text
                    )
                )
            else:
                decisions.append(self._entity_to_decision_heuristic(field_name, best_entity))
        return decisions

    def _entity_to_decision_heuristic(
        self, field_name: str, entity: ExtractedEntity
    ) -> FieldDecision:
        """Score and route using the heuristic FieldConfidenceScorer."""
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
            scorer="heuristic",
        )

    def _entity_to_decision_rf(
        self, field_name: str, entity: ExtractedEntity, document_text: str
    ) -> FieldDecision:
        """Score and route using the RF confidence model.

        Falls back to the heuristic path if the RF model raises
        ``RFModelNotLoadedError`` (e.g. model file was deleted between
        startup and this call).

        Args:
            field_name: NER label for this entity.
            entity: The best extracted entity for this field.
            document_text: Full source document text for feature building.

        Returns:
            A ``FieldDecision`` annotated with ``scorer="rf"``.
        """
        try:
            prediction = self.rf_model.predict(entity, document_text)  # type: ignore[union-attr]
            return FieldDecision(
                field_name=field_name,
                value=entity.text,
                confidence=prediction.confidence,
                decision=prediction.decision,
                sources=list(entity.sources),
                confidence_factors={"rf_confidence": prediction.confidence},
                start=entity.start,
                end=entity.end,
                scorer="rf",
            )
        except RFModelNotLoadedError:
            logger.warning(
                "RF model not loaded during pipeline run for field=%s; "
                "falling back to heuristic scorer.",
                field_name,
            )
            return self._entity_to_decision_heuristic(field_name, entity)

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


# ──────────────────────────────────────────────────────────────────────────────
# RF model loader utility (used at API startup and in retraining pipeline)
# ──────────────────────────────────────────────────────────────────────────────


class RFModelLoader:
    """Discover and load the latest versioned RF confidence model from disk.

    The loader scans ``rf_model_output_dir`` for files matching
    ``rf_confidence_v*.joblib``, picks the newest by filename
    (filenames are timestamp-stamped so lexicographic order = time order),
    and loads it.

    Usage (FastAPI lifespan or CLI)::

        loader = RFModelLoader()
        rf_model = loader.load_latest()   # None if no model exists yet
        if rf_model is not None:
            engine.switch_to_rf_model(rf_model)

    Args:
        settings: Optional settings override.
        use_bert: Passed to ``RFConfidenceModel``; set ``False`` in tests.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        use_bert: bool = True,
    ) -> None:
        """Initialise the loader.

        Args:
            settings: Optional settings override.
            use_bert: Whether to enable BERT embeddings in the loaded model.
        """
        self.settings = settings or get_settings()
        self.use_bert = use_bert

    def load_latest(self) -> Optional[RFConfidenceModel]:
        """Load the most recently trained RF model, or return ``None``.

        Returns:
            A loaded ``RFConfidenceModel`` instance, or ``None`` if no
            model file exists yet (e.g. before first training run).
        """
        model_dir = Path(self.settings.rf_model_output_dir)
        if not model_dir.exists():
            logger.info("RF model directory does not exist yet: %s.", model_dir)
            return None

        candidates = sorted(model_dir.glob("rf_confidence_v*.joblib"), reverse=True)
        if not candidates:
            logger.info("No trained RF model found in %s.", model_dir)
            return None

        latest_path = candidates[0]
        rf_model = RFConfidenceModel(settings=self.settings, use_bert=self.use_bert)
        try:
            rf_model.load(latest_path)
            logger.info("Auto-loaded RF model: %s.", latest_path.name)
            return rf_model
        except Exception as exc:
            logger.error("Failed to load RF model from %s: %s.", latest_path, exc)
            return None

    def list_available(self) -> List[str]:
        """Return filenames of all available versioned RF models.

        Returns:
            Sorted list of ``.joblib`` filenames, newest first.
        """
        model_dir = Path(self.settings.rf_model_output_dir)
        if not model_dir.exists():
            return []
        return [
            candidate.name
            for candidate in sorted(model_dir.glob("rf_confidence_v*.joblib"), reverse=True)
        ]
