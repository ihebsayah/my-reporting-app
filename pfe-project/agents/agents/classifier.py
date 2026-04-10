"""Agent 1: Document Classifier Agent.

Determines the type of the document (invoice, receipt, contract, report,
letter, unknown) using:
1. BART zero-shot classifier (via ML tools → FastAPI pipeline).
2. Redis pattern lookup (has this doc type been seen before?).
3. NLP heuristics (keyword density, document length, field presence).

ReAct loop: Think → Use tool → Observe → Decide.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agents.config import get_agent_settings
from agents.memory.short_term import ShortTermMemory
from agents.tools.ml_tools import run_bart_classification, run_ner_extraction

logger = logging.getLogger(__name__)
_settings = get_agent_settings()


@dataclass
class ClassifierResult:
    """Output of the Document Classifier Agent."""

    doc_type: str
    confidence: float
    reasoning: str
    field_names: List[str] = field(default_factory=list)
    raw_scores: Dict[str, float] = field(default_factory=dict)


class ClassifierAgent:
    """Document Classifier Agent — determines document type via ReAct reasoning.

    This agent wraps the existing BART classifier and enriches its output with
    NER field signals and Redis-cached document-type patterns.

    Args:
        memory: Optional shared short-term memory for this extraction run.

    Example::

        agent = ClassifierAgent()
        result = agent.run("Invoice #123 from Acme Corp for $5000")
        print(result.doc_type, result.confidence)
    """

    AGENT_NAME = "classifier"

    def __init__(self, memory: Optional[ShortTermMemory] = None) -> None:
        """Initialise the classifier agent.

        Args:
            memory: Shared short-term memory instance for this document run.
        """
        self.memory = memory
        logger.info("ClassifierAgent initialised.")

    def run(self, text: str) -> ClassifierResult:
        """Classify the document type using a ReAct-style reasoning loop.

        Step 1: Think — what signals will help me classify this document?
        Step 2: Act — call NER extraction to get field labels.
        Step 3: Observe — what fields are present?
        Step 4: Act — call BART classification.
        Step 5: Observe — what does BART say?
        Step 6: Decide — combine signals and return final classification.

        Args:
            text: Raw document text.

        Returns:
            ClassifierResult with doc_type, confidence, and reasoning.
        """
        logger.info("ClassifierAgent.run started (text_len=%d).", len(text))

        # ── Step 1: Think ─────────────────────────────────────────────────
        self._think(
            "I need to classify this document. I'll first extract NER fields "
            "to understand the field composition, then run the BART classifier."
        )

        # ── Step 2 & 3: Extract NER fields ────────────────────────────────
        ner_raw = run_ner_extraction.invoke({"text": text})
        ner_data = self._parse_json(ner_raw, {})
        entities = ner_data.get("entities", [])
        field_names = list({e.get("label", "") for e in entities if e.get("label")})
        self._observe(f"NER extracted {len(entities)} entities: labels={field_names}")

        # ── Step 4 & 5: Run BART classification ───────────────────────────
        bart_raw = run_bart_classification.invoke({"text": text})
        bart_data = self._parse_json(bart_raw, {})
        bart_type = bart_data.get("doc_type", "unknown")
        bart_confidence = bart_data.get("confidence", 0.4)
        self._observe(f"BART says: doc_type={bart_type}, confidence={bart_confidence:.3f}")

        # ── Step 6: Decide ────────────────────────────────────────────────
        result = self._decide(
            bart_type=bart_type,
            bart_confidence=bart_confidence,
            field_names=field_names,
            text=text,
        )

        # Persist to short-term memory.
        if self.memory:
            self.memory.set_context("classifier_result", {
                "doc_type": result.doc_type,
                "confidence": result.confidence,
                "reasoning": result.reasoning,
                "field_names": result.field_names,
            })
            self.memory.add_message("ai", result.reasoning, agent=self.AGENT_NAME)

        logger.info(
            "ClassifierAgent.run complete: doc_type=%s confidence=%.3f",
            result.doc_type, result.confidence,
        )
        return result

    # ── ReAct helpers ──────────────────────────────────────────────────────

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

    def _decide(
        self,
        bart_type: str,
        bart_confidence: float,
        field_names: List[str],
        text: str,
    ) -> ClassifierResult:
        """Combine BART output with field composition to produce final classification."""
        doc_type = bart_type
        confidence = bart_confidence

        # Boost confidence when NER field composition strongly agrees.
        has_invoice_id = "INVOICE_ID" in field_names
        has_amount = "TOTAL_AMOUNT" in field_names
        has_vendor = "VENDOR_NAME" in field_names
        has_date = "INVOICE_DATE" in field_names

        if doc_type == "invoice" and has_invoice_id and has_amount and has_vendor:
            confidence = min(0.99, confidence + 0.08)
            reasoning = (
                f"BART classified as '{doc_type}' (base conf={bart_confidence:.2f}). "
                f"NER confirmed: INVOICE_ID={has_invoice_id}, TOTAL_AMOUNT={has_amount}, "
                f"VENDOR_NAME={has_vendor}, INVOICE_DATE={has_date}. "
                f"Field composition strongly confirms invoice. Final confidence={confidence:.2f}."
            )
        elif doc_type == "unknown" or confidence < 0.5:
            # Fallback heuristic.
            if has_invoice_id or (has_amount and has_vendor):
                doc_type, confidence = "invoice", 0.72
                reasoning = (
                    "BART was uncertain. NER found invoice-like fields "
                    f"(INVOICE_ID={has_invoice_id}, TOTAL_AMOUNT={has_amount}, "
                    f"VENDOR={has_vendor}). Reclassifying as 'invoice' with confidence=0.72."
                )
            else:
                doc_type, confidence = "unknown", 0.40
                reasoning = (
                    "BART returned low-confidence result and NER fields are ambiguous. "
                    "Classifying as 'unknown' — will require human review."
                )
        else:
            reasoning = (
                f"BART classified as '{doc_type}' with confidence {bart_confidence:.2f}. "
                f"NER fields present: {field_names}. Classification accepted."
            )

        return ClassifierResult(
            doc_type=doc_type,
            confidence=round(confidence, 4),
            reasoning=reasoning,
            field_names=field_names,
        )

    @staticmethod
    def _parse_json(raw: str, default: Any) -> Any:
        """Safely parse a JSON string, returning default on failure."""
        try:
            return json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            return default
