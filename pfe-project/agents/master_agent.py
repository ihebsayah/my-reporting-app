"""Master Agent — Extraction Supervisor.

The Master Agent orchestrates the full extraction workflow:
1. Initialises a ShortTermMemory session for this document.
2. Calls ClassifierAgent → ExtractorAgent → ValidatorAgent → RouterAgent in sequence.
3. Assembles the final response with full reasoning chain.
4. Persists the decision to the DB and triggers real-time learning.
5. On failure, falls back to the existing ML pipeline decision.

This is the single entry point for agent-assisted extraction.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from agents.agents.classifier import ClassifierAgent
from agents.agents.extractor import ExtractorAgent
from agents.agents.router import AUTO_APPROVE, HUMAN_REVIEW, REJECT, RouterAgent
from agents.agents.validator import ValidatorAgent
from agents.config import get_agent_settings
from agents.memory.long_term import LongTermMemory
from agents.memory.short_term import ShortTermMemory
from agents.tools.db_tools import save_agent_decision

logger = logging.getLogger(__name__)
_settings = get_agent_settings()


@dataclass
class AgentExtractionResult:
    """Full result of a Master Agent extraction run — returned to FastAPI."""

    document_id: str
    action: str                  # "auto_approve" | "human_review" | "reject"
    confidence: float
    agent_reasoning: str         # Human-readable explanation for the dashboard
    doc_type: str
    extracted_fields: List[Dict[str, Any]]
    validation_issues: List[Dict[str, Any]]
    safety_rails_triggered: List[str]
    session_id: str
    duration_ms: int
    agents_used: List[str]
    fallback_used: bool = False
    error: Optional[str] = None


class MasterAgent:
    """Extraction Supervisor — orchestrates the 4 sub-agents sequentially.

    Follows the plan's 16-step workflow and handles:
    - Sub-agent orchestration (sequential calls).
    - Short-term session memory (per document).
    - Long-term learning (post-decision).
    - Graceful fallback on any sub-agent failure.
    - Decision persistence for monitoring.

    Usage::

        master = MasterAgent()
        result = master.process_document("doc_001", "Invoice #123 from Acme for $5000")
        print(result.action, result.agent_reasoning)
    """

    def __init__(self) -> None:
        """Initialise the Master Agent and its sub-agents."""
        if not _settings.agents_enabled:
            logger.warning("Agents are GLOBALLY DISABLED. MasterAgent will use fallback only.")
        self.ltm = LongTermMemory()
        logger.info("MasterAgent initialised.")

    def process_document(
        self,
        document_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentExtractionResult:
        """Process a document through all 4 sub-agents and return a final decision.

        Args:
            document_id: Unique document identifier (used for DB persistence).
            text: Raw document text.
            metadata: Optional extra metadata passed from FastAPI.

        Returns:
            AgentExtractionResult with action, reasoning, and all agent outputs.
        """
        start_ms = time.time()
        logger.info(
            "MasterAgent.process_document started: document_id=%s text_len=%d.",
            document_id, len(text),
        )

        # ── Global disable check ───────────────────────────────────────────
        if not _settings.agents_enabled:
            return self._build_fallback_result(
                document_id, text, "Agents globally disabled.", start_ms
            )

        # ── Initialise session memory ─────────────────────────────────────
        memory = ShortTermMemory(document_id=document_id)
        memory.add_message(
            "human",
            f"Process document: {document_id} (len={len(text)})",
        )

        agents_used: List[str] = []

        try:
            # ── Step 1: Classify ───────────────────────────────────────────
            logger.info("[MASTER] Calling ClassifierAgent.")
            classifier = ClassifierAgent(memory=memory)
            classifier_result = classifier.run(text)
            agents_used.append("classifier")
            memory.set_context("classifier_done", True)

            # ── Step 2: Extract ────────────────────────────────────────────
            logger.info("[MASTER] Calling ExtractorAgent.")
            extractor = ExtractorAgent(memory=memory)
            extractor_result = extractor.run(text, doc_type=classifier_result.doc_type)
            agents_used.append("extractor")
            memory.set_context("extractor_done", True)

            # Prepare extracted field dicts for downstream agents.
            field_dicts = [
                {
                    "field_name": f.field_name,
                    "value": f.value,
                    "confidence": f.confidence,
                    "sources": f.sources,
                    "decision": f.decision,
                    "start": f.start,
                    "end": f.end,
                }
                for f in extractor_result.fields
            ]

            # ── Step 3: Validate ───────────────────────────────────────────
            logger.info("[MASTER] Calling ValidatorAgent.")
            validator = ValidatorAgent(memory=memory, long_term_memory=self.ltm)
            validator_result = validator.run(field_dicts, doc_type=classifier_result.doc_type)
            agents_used.append("validator")
            memory.set_context("validator_done", True)

            # Convert validation issues to dicts for serialisation.
            issue_dicts = [
                {
                    "field_name": issue.field_name,
                    "issue_type": issue.issue_type,
                    "severity": issue.severity,
                    "description": issue.description,
                }
                for issue in validator_result.issues
            ]

            # Prepare validation summary dict for RouterAgent.
            validation_summary = {
                "is_valid": validator_result.is_valid,
                "confidence_adjustment": validator_result.confidence_adjustment,
                "vendor_known": validator_result.vendor_known,
                "amount_normal": validator_result.amount_normal,
                "date_valid": validator_result.date_valid,
                "issues": issue_dicts,
            }

            # ── Step 4: Route ──────────────────────────────────────────────
            logger.info("[MASTER] Calling RouterAgent.")
            router = RouterAgent(memory=memory, long_term_memory=self.ltm)
            router_result = router.run(
                extraction_confidence=extractor_result.overall_confidence,
                validation_result=validation_summary,
                extracted_fields=field_dicts,
                doc_type=classifier_result.doc_type,
            )
            agents_used.append("router")

            # ── Assemble final reasoning ───────────────────────────────────
            agent_reasoning = self._build_master_reasoning(
                document_id=document_id,
                doc_type=classifier_result.doc_type,
                classifier_reasoning=classifier_result.reasoning,
                extractor_reasoning=extractor_result.reasoning,
                validator_reasoning=validator_result.reasoning,
                router_reasoning=router_result.reasoning,
                action=router_result.action,
                confidence=router_result.confidence,
            )
            memory.add_message("ai", agent_reasoning, agent="master")

            # ── Persist decision ───────────────────────────────────────────
            save_agent_decision(
                document_id=document_id,
                action=router_result.action,
                confidence=router_result.confidence,
                reasoning=agent_reasoning,
            )

            # ── Persist session to Redis for audit ─────────────────────────
            memory.persist_to_redis()

            duration_ms = int((time.time() - start_ms) * 1000)
            logger.info(
                "MasterAgent.process_document complete: document_id=%s action=%s "
                "confidence=%.3f duration_ms=%d.",
                document_id, router_result.action, router_result.confidence, duration_ms,
            )

            return AgentExtractionResult(
                document_id=document_id,
                action=router_result.action,
                confidence=router_result.confidence,
                agent_reasoning=agent_reasoning,
                doc_type=classifier_result.doc_type,
                extracted_fields=field_dicts,
                validation_issues=issue_dicts,
                safety_rails_triggered=router_result.safety_rails_triggered,
                session_id=memory.session_id,
                duration_ms=duration_ms,
                agents_used=agents_used,
                fallback_used=False,
            )

        except Exception as exc:
            logger.exception(
                "MasterAgent.process_document FAILED for document_id=%s: %s.",
                document_id, exc,
                exc_info=exc,
            )
            memory.add_message("ai", f"[ERROR] {exc}", agent="master")
            memory.persist_to_redis()
            return self._build_fallback_result(document_id, text, str(exc), start_ms)

    def record_human_feedback(
        self,
        document_id: str,
        human_outcome: str,
        agent_decision: str,
        vendor: str = "",
        amount: float = 0.0,
        notes: Optional[str] = None,
    ) -> None:
        """Record human feedback for real-time learning.

        Called after a human approves/rejects/overrides an agent decision.
        Updates both Redis (immediate) and PostgreSQL (durable) patterns.

        Args:
            document_id: The document that was processed.
            human_outcome: Human's final decision (approve/reject/review).
            agent_decision: What the agent originally decided.
            vendor: Extracted vendor name (for pattern storage).
            amount: Extracted invoice amount.
            notes: Optional human reviewer notes.
        """
        logger.info(
            "MasterAgent.record_human_feedback: doc=%s human=%s agent=%s vendor=%s.",
            document_id, human_outcome, agent_decision, vendor,
        )

        # Update the DB agent_decision record with the human override.
        try:
            from agents.tools.db_tools import _get_session
            from sqlalchemy import text

            session = _get_session()
            session.execute(
                text(
                    "UPDATE agent_decisions SET human_override = :override, human_feedback = :notes "
                    "WHERE document_id = :doc_id ORDER BY id DESC LIMIT 1"
                ),
                {"override": human_outcome, "notes": notes, "doc_id": document_id},
            )
            session.commit()
            session.close()
        except Exception as exc:
            logger.warning("Could not update agent_decisions with human override: %s", exc)

        # Record learning signal in long-term memory.
        if vendor:
            self.ltm.record_feedback(
                document_id=document_id,
                vendor=vendor,
                amount=amount,
                agent_decision=agent_decision,
                human_outcome=human_outcome,
                notes=notes,
            )

    # ── Private helpers ────────────────────────────────────────────────────

    def _build_master_reasoning(
        self,
        document_id: str,
        doc_type: str,
        classifier_reasoning: str,
        extractor_reasoning: str,
        validator_reasoning: str,
        router_reasoning: str,
        action: str,
        confidence: float,
    ) -> str:
        """Build a single human-readable reasoning string for the dashboard."""
        action_emoji = {"auto_approve": "✅", "human_review": "🔍", "reject": "❌"}.get(action, "❓")
        return (
            f"{action_emoji} Document '{document_id}' — Final Decision: {action.upper().replace('_', ' ')} "
            f"(confidence={confidence:.1%})\n\n"
            f"1. CLASSIFIER: {classifier_reasoning}\n\n"
            f"2. EXTRACTOR: {extractor_reasoning}\n\n"
            f"3. VALIDATOR: {validator_reasoning}\n\n"
            f"4. ROUTER: {router_reasoning}"
        )

    def _build_fallback_result(
        self,
        document_id: str,
        text: str,
        error_msg: str,
        start_ms: float,
    ) -> AgentExtractionResult:
        """Return a safe fallback result using the existing pipeline's default."""
        duration_ms = int((time.time() - start_ms) * 1000)
        logger.warning(
            "MasterAgent fallback activated for document_id=%s: %s", document_id, error_msg
        )
        return AgentExtractionResult(
            document_id=document_id,
            action=HUMAN_REVIEW,  # Safe default: always route to human on failure.
            confidence=0.0,
            agent_reasoning=(
                f"⚠️ Agent system encountered an error and fell back to safe default (human_review). "
                f"Error: {error_msg}. "
                "The existing ML pipeline decision is unaffected."
            ),
            doc_type="unknown",
            extracted_fields=[],
            validation_issues=[],
            safety_rails_triggered=[],
            session_id="fallback",
            duration_ms=duration_ms,
            agents_used=[],
            fallback_used=True,
            error=error_msg,
        )
