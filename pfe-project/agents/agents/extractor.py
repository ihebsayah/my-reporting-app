"""Agent 2: Field Extractor Agent.

Given a document type from the Classifier Agent, this agent:
1. Runs the spaCy + regex NER ensemble (via ML tools).
2. Runs the Random Forest confidence scorer.
3. Returns structured, typed field objects with per-field confidence.

ReAct loop: Think → Extract with NER → Observe → Score → Observe → Return.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agents.config import get_agent_settings
from agents.llm_reasoner import get_reasoner
from agents.memory.short_term import ShortTermMemory
from agents.tools.ml_tools import run_ner_extraction, run_confidence_scoring

logger = logging.getLogger(__name__)
_settings = get_agent_settings()


@dataclass
class ExtractedField:
    """One extracted field with value and confidence."""

    field_name: str
    value: str
    confidence: float
    sources: List[str] = field(default_factory=list)
    decision: str = "review"  # "auto" | "review" | "reject"
    start: Optional[int] = None
    end: Optional[int] = None


@dataclass
class ExtractorResult:
    """Output of the Field Extractor Agent."""

    fields: List[ExtractedField]
    overall_confidence: float
    reasoning: str
    doc_type: str
    raw_pipeline_decision: Optional[str] = None


class ExtractorAgent:
    """Field Extractor Agent — extracts and scores document fields.

    Wraps the existing spaCy NER + Random Forest pipeline endpoints as tools
    and returns enriched field objects to the Master Agent.

    Args:
        memory: Optional shared short-term memory for this extraction run.
    """

    AGENT_NAME = "extractor"

    def __init__(self, memory: Optional[ShortTermMemory] = None) -> None:
        """Initialise the extractor agent."""
        self.memory = memory
        logger.info("ExtractorAgent initialised.")

    def run(self, text: str, doc_type: str = "invoice") -> ExtractorResult:
        """Extract fields from document text using the existing ML pipeline.

        ReAct steps:
        1. Think — what fields should I look for given this doc type?
        2. Act — run the full pipeline (NER + RF confidence scorer).
        3. Observe — what fields were found with what confidence?
        4. Act — if any critical fields are missing, try NER-only extraction.
        5. Decide — compile and return fields with reasoning.

        Args:
            text: Raw document text.
            doc_type: Document type from ClassifierAgent (used for reasoning).

        Returns:
            ExtractorResult with per-field data and overall confidence.
        """
        logger.info(
            "ExtractorAgent.run started: doc_type=%s text_len=%d.", doc_type, len(text)
        )

        # ── Think ─────────────────────────────────────────────────────────
        target_fields = self._target_fields_for_type(doc_type)
        self._think(
            f"Document is a {doc_type}. I'll run the full pipeline to extract "
            f"target fields: {target_fields}."
        )

        # ── Act: Full pipeline (NER + RF confidence) ──────────────────────
        pipeline_raw = run_confidence_scoring.invoke({"text": text})
        pipeline_data = self._parse_json(pipeline_raw, {})
        pipeline_fields = pipeline_data.get("fields", [])
        overall_pipeline_decision = pipeline_data.get("overall_decision", "review")
        self._observe(
            f"Pipeline returned {len(pipeline_fields)} fields, "
            f"overall_decision={overall_pipeline_decision}."
        )

        # ── Act: Supplementary NER extraction if fields are missing ───────
        found_labels = {f.get("field_name", "") for f in pipeline_fields}
        missing = [t for t in target_fields if t not in found_labels]
        ner_entities: List[Dict[str, Any]] = []
        if missing:
            self._think(f"Missing fields: {missing}. Running NER-only extraction.")
            ner_raw = run_ner_extraction.invoke({"text": text})
            ner_data = self._parse_json(ner_raw, {})
            ner_entities = ner_data.get("entities", [])
            self._observe(f"NER supplementary extraction: {len(ner_entities)} entities.")

        # ── Decide: Merge pipeline + supplementary NER, then LLM review ────
        extracted_fields = self._merge_fields(pipeline_fields, ner_entities, target_fields)

        # ── LLM review (optional upgrade) ─────────────────────────────────
        extracted_fields = self._llm_review_fields(
            doc_type=doc_type,
            extracted_fields=extracted_fields,
            missing_fields=missing,
        )

        overall_confidence = self._compute_overall_confidence(extracted_fields)

        reasoning = self._build_reasoning(
            doc_type, extracted_fields, overall_confidence, overall_pipeline_decision, missing
        )

        result = ExtractorResult(
            fields=extracted_fields,
            overall_confidence=round(overall_confidence, 4),
            reasoning=reasoning,
            doc_type=doc_type,
            raw_pipeline_decision=overall_pipeline_decision,
        )

        # Persist to shared memory.
        if self.memory:
            self.memory.set_context("extractor_result", {
                "fields": [
                    {
                        "field_name": f.field_name,
                        "value": f.value,
                        "confidence": f.confidence,
                        "decision": f.decision,
                    }
                    for f in extracted_fields
                ],
                "overall_confidence": overall_confidence,
                "reasoning": reasoning,
            })
            self.memory.add_message("ai", reasoning, agent=self.AGENT_NAME)

        logger.info(
            "ExtractorAgent.run complete: %d fields, overall_conf=%.3f.",
            len(extracted_fields), overall_confidence,
        )
        return result

    # ── Internal helpers ───────────────────────────────────────────────────

    def _llm_review_fields(
        self,
        doc_type: str,
        extracted_fields: List[ExtractedField],
        missing_fields: List[str],
    ) -> List[ExtractedField]:
        """Ask the LLM which extracted fields are reliable; adjust decisions.

        If the LLM is unavailable or returns unparseable output the original
        field list is returned unchanged.

        Args:
            doc_type: Document type.
            extracted_fields: Fields from pipeline + NER merge.
            missing_fields: Target fields not found.

        Returns:
            Possibly-updated list of ExtractedField objects.
        """
        reasoner = get_reasoner()
        if not reasoner.is_available():
            self._think("LLM not available — keeping heuristic field decisions.")
            return extracted_fields

        self._think("LLM available — reviewing field reliability.")
        field_dicts = [
            {"field_name": f.field_name, "value": f.value, "confidence": f.confidence}
            for f in extracted_fields
        ]
        llm_output = reasoner.review_extraction(
            doc_type=doc_type,
            fields=field_dicts,
            missing_fields=missing_fields,
        )
        if not llm_output:
            self._observe("LLM extraction review returned no output — using heuristic fields.")
            return extracted_fields

        reliable = set(llm_output.get("reliable_fields", []))
        uncertain = set(llm_output.get("uncertain_fields", []))
        llm_reason = llm_output.get("reasoning", "")
        self._observe(
            f"LLM review: reliable={sorted(reliable)}, uncertain={sorted(uncertain)}. "
            f"Reason: {llm_reason}"
        )

        # Apply LLM reliability signals to field decisions.
        for f in extracted_fields:
            if f.field_name in reliable:
                f.decision = "auto" if f.confidence >= 0.75 else "review"
            elif f.field_name in uncertain:
                f.decision = "review"
                f.confidence = max(0.0, f.confidence - 0.05)  # Small penalty.
        return extracted_fields

    def _target_fields_for_type(self, doc_type: str) -> List[str]:
        """Return expected field labels for a given document type."""
        type_fields = {
            "invoice": ["INVOICE_ID", "INVOICE_DATE", "TOTAL_AMOUNT", "VENDOR_NAME"],
            "receipt": ["TOTAL_AMOUNT", "INVOICE_DATE"],
            "contract": ["VENDOR_NAME", "INVOICE_DATE"],
            "report": ["INVOICE_DATE"],
        }
        return type_fields.get(doc_type, ["INVOICE_ID", "INVOICE_DATE", "TOTAL_AMOUNT", "VENDOR_NAME"])

    def _merge_fields(
        self,
        pipeline_fields: List[Dict[str, Any]],
        ner_entities: List[Dict[str, Any]],
        target_fields: List[str],
    ) -> List[ExtractedField]:
        """Merge pipeline fields with NER supplementary entities."""
        merged: Dict[str, ExtractedField] = {}

        # Primary: pipeline fields (RF confidence scored).
        for f in pipeline_fields:
            name = f.get("field_name", "")
            merged[name] = ExtractedField(
                field_name=name,
                value=str(f.get("value", "")),
                confidence=float(f.get("confidence", 0.0)),
                sources=list(f.get("sources", [])),
                decision=f.get("decision", "review"),
                start=f.get("start"),
                end=f.get("end"),
            )

        # Supplementary: NER entities for any still-missing target fields.
        labels_found = set(merged.keys())
        for entity in ner_entities:
            label = entity.get("label", "")
            if label in target_fields and label not in labels_found:
                merged[label] = ExtractedField(
                    field_name=label,
                    value=str(entity.get("text", "")),
                    confidence=float(entity.get("score", 0.6)),
                    sources=list(entity.get("sources", ["ner_supplementary"])),
                    decision="review",
                    start=entity.get("start"),
                    end=entity.get("end"),
                )

        return list(merged.values())

    def _compute_overall_confidence(self, fields: List[ExtractedField]) -> float:
        """Average per-field confidence, weighted by field importance."""
        if not fields:
            return 0.0
        weights = {
            "INVOICE_ID": 1.2,
            "TOTAL_AMOUNT": 1.5,
            "VENDOR_NAME": 1.3,
            "INVOICE_DATE": 1.0,
        }
        total_weight = 0.0
        weighted_sum = 0.0
        for f in fields:
            w = weights.get(f.field_name, 1.0)
            weighted_sum += f.confidence * w
            total_weight += w
        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _build_reasoning(
        self,
        doc_type: str,
        fields: List[ExtractedField],
        overall_confidence: float,
        pipeline_decision: str,
        missing_fields: List[str],
    ) -> str:
        """Generate a human-readable extraction reasoning string."""
        field_summary = ", ".join(
            f"{f.field_name}={f.value!r}(conf={f.confidence:.2f})" for f in fields
        )
        missing_note = (
            f" Missing fields: {missing_fields}." if missing_fields else " All target fields found."
        )
        return (
            f"Extracted {len(fields)} fields from a '{doc_type}' document. "
            f"Fields: [{field_summary}].{missing_note} "
            f"Overall confidence={overall_confidence:.3f}. "
            f"Pipeline baseline decision: {pipeline_decision}."
        )

    def _think(self, thought: str) -> None:
        if _settings.log_agent_reasoning:
            logger.info("[%s THINK] %s", self.AGENT_NAME.upper(), thought)
        if self.memory:
            self.memory.add_message("ai", f"[THINK] {thought}", agent=self.AGENT_NAME)

    def _observe(self, observation: str) -> None:
        if _settings.log_agent_reasoning:
            logger.info("[%s OBSERVE] %s", self.AGENT_NAME.upper(), observation)
        if self.memory:
            self.memory.add_message("tool", f"[OBSERVE] {observation}", agent=self.AGENT_NAME)

    @staticmethod
    def _parse_json(raw: str, default: Any) -> Any:
        try:
            return json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            return default
