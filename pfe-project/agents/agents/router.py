"""Agent 4: Router Agent.

Makes the final routing decision: auto_approve | human_review | reject.

Decision logic combines:
1. Extraction confidence (from ExtractorAgent).
2. Validation result (from ValidatorAgent).
3. Safety rails (always enforced before any other check).
4. Vendor trustworthiness from Redis patterns (LongTermMemory).
5. Historical override rate for similar cases.

Safety rails always take priority — no LLM reasoning can bypass them.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agents.config import get_agent_settings
from agents.llm_reasoner import get_reasoner
from agents.memory.long_term import LongTermMemory
from agents.memory.short_term import ShortTermMemory

logger = logging.getLogger(__name__)
_settings = get_agent_settings()

# Routing action constants.
AUTO_APPROVE = "auto_approve"
HUMAN_REVIEW = "human_review"
REJECT = "reject"


@dataclass
class RouterResult:
    """Output of the Router Agent."""

    action: str          # "auto_approve" | "human_review" | "reject"
    confidence: float    # Final adjusted confidence score used for decision
    reasoning: str
    safety_rails_triggered: List[str] = field(default_factory=list)
    signals: Dict[str, Any] = field(default_factory=dict)


class RouterAgent:
    """Router Agent — makes the final routing decision with full reasoning.

    Applies safety rails FIRST (non-negotiable), then uses configurable
    thresholds and contextual signals to choose the routing action.

    Args:
        memory: Optional shared short-term memory.
        long_term_memory: Optional long-term memory for pattern context.
    """

    AGENT_NAME = "router"

    def __init__(
        self,
        memory: Optional[ShortTermMemory] = None,
        long_term_memory: Optional[LongTermMemory] = None,
    ) -> None:
        """Initialise the Router Agent."""
        self.memory = memory
        self.ltm = long_term_memory or LongTermMemory()
        logger.info("RouterAgent initialised.")

    def run(
        self,
        extraction_confidence: float,
        validation_result: Dict[str, Any],
        extracted_fields: List[Dict[str, Any]],
        doc_type: str = "invoice",
    ) -> RouterResult:
        """Determine routing action from all agent signals.

        ReAct steps:
        1. Think — what signals do I have? What are the thresholds?
        2. Act — check safety rails (non-negotiable).
        3. Observe — are any rails triggered?
        4. Act — apply confidence adjustment from validator.
        5. Observe — what is the final confidence?
        6. Act — check Redis for vendor trustworthiness.
        7. Observe — does the vendor pattern support or oppose auto-approve?
        8. Decide — final routing decision with full reasoning.

        Args:
            extraction_confidence: Overall confidence from ExtractorAgent.
            validation_result: Dict output from ValidatorAgent.
            extracted_fields: List of field dicts from ExtractorAgent.
            doc_type: Document type for contextual reasoning.

        Returns:
            RouterResult with final action and reasoning.
        """
        logger.info(
            "RouterAgent.run started: extraction_conf=%.3f, doc_type=%s.",
            extraction_confidence, doc_type,
        )

        # Index fields by name.
        field_map: Dict[str, str] = {
            f.get("field_name", ""): str(f.get("value", ""))
            for f in extracted_fields
        }

        # ── Think ─────────────────────────────────────────────────────────
        self._think(
            f"I have extraction_confidence={extraction_confidence:.3f}, "
            f"validation_is_valid={validation_result.get('is_valid', False)}, "
            f"vendor_known={validation_result.get('vendor_known', False)}. "
            f"Auto-approve threshold={_settings.auto_approve_threshold}, "
            f"human-review threshold={_settings.human_review_threshold}. "
            "Let me check safety rails first."
        )

        # ── Safety rails (unconditional priority) ────────────────────────
        rails_triggered: List[str] = []
        force_action: Optional[str] = None

        rails_triggered, force_action = self._check_safety_rails(
            validation_result, field_map, extraction_confidence
        )

        if rails_triggered:
            self._observe(f"Safety rails triggered: {rails_triggered}. Action forced to {force_action}.")
        else:
            self._observe("No safety rails triggered. Proceeding to confidence-based routing.")

        if force_action:
            reasoning = self._build_reasoning(
                force_action, extraction_confidence, validation_result,
                rails_triggered, field_map, is_forced=True
            )
            result = RouterResult(
                action=force_action,
                confidence=extraction_confidence,
                reasoning=reasoning,
                safety_rails_triggered=rails_triggered,
                signals=self._build_signals(extraction_confidence, validation_result, field_map),
            )
            return self._finalise(result)

        # ── Apply confidence adjustment from validator ─────────────────────
        adj = float(validation_result.get("confidence_adjustment", 0.0))
        adjusted_confidence = min(0.99, max(0.0, extraction_confidence + adj))
        self._observe(
            f"Confidence after validation adjustment ({adj:+.3f}): {adjusted_confidence:.3f}."
        )

        # ── Vendor trustworthiness signal ─────────────────────────────────
        vendor = field_map.get("VENDOR_NAME", "")
        vendor_pattern = self.ltm.lookup_vendor_pattern(vendor) if vendor else None
        vendor_trust_boost = 0.0
        if vendor_pattern:
            total = vendor_pattern.get("total", 0)
            if total > 5:
                approve_ct = vendor_pattern.get("approve", 0)
                reject_ct = vendor_pattern.get("reject", 0)
                approve_rate = approve_ct / total if total > 0 else 0
                reject_rate = reject_ct / total if total > 0 else 0
                if approve_rate > 0.8:
                    vendor_trust_boost = 0.03
                    self._observe(
                        f"Vendor '{vendor}' has high approval rate ({approve_rate:.0%}) — boosting confidence."
                    )
                elif reject_rate > 0.5:
                    vendor_trust_boost = -0.05
                    self._observe(
                        f"Vendor '{vendor}' has high rejection rate ({reject_rate:.0%}) — penalising confidence."
                    )

        final_confidence = min(0.99, max(0.0, adjusted_confidence + vendor_trust_boost))
        self._observe(f"Final adjusted confidence: {final_confidence:.3f}.")

        # ── Routing decision ───────────────────────────────────────────────
        is_valid = bool(validation_result.get("is_valid", False))
        vendor_known = bool(validation_result.get("vendor_known", False))

        # Try LLM routing first — falls back to heuristic automatically.
        action = self._decide_with_llm(
            doc_type=doc_type,
            final_confidence=final_confidence,
            is_valid=is_valid,
            vendor_known=vendor_known,
            field_map=field_map,
            rails_triggered=rails_triggered,
            validation_result=validation_result,
        )
        self._observe(f"Routing decision: {action}.")

        reasoning = self._build_reasoning(
            action, final_confidence, validation_result,
            rails_triggered, field_map, is_forced=False,
            vendor_trust_boost=vendor_trust_boost,
        )

        result = RouterResult(
            action=action,
            confidence=round(final_confidence, 4),
            reasoning=reasoning,
            safety_rails_triggered=rails_triggered,
            signals=self._build_signals(final_confidence, validation_result, field_map),
        )
        return self._finalise(result)

    # ── Safety rail enforcement ────────────────────────────────────────────

    def _check_safety_rails(
        self,
        validation_result: Dict[str, Any],
        field_map: Dict[str, str],
        confidence: float,
    ) -> tuple:
        """Check all safety rails and return (triggered_list, forced_action).

        Safety rails, in priority order:
        1. Amount > $100,000 → human_review (not auto_approve).
        2. New vendor (not in DB) → human_review.
        3. Anomaly detected (validator error) → human_review.
        4. Confidence < 0.40 (very low) → reject.
        """
        rails: List[str] = []
        force: Optional[str] = None

        issues = validation_result.get("issues", [])

        # Rail 1: Amount limit.
        amount_str = field_map.get("TOTAL_AMOUNT", "")
        import re
        cleaned = re.sub(r"[^\d.]", "", amount_str.replace(",", "")) if amount_str else ""
        try:
            amount = float(cleaned)
            if amount > _settings.safety_max_amount:
                rails.append(f"RAIL_1_AMOUNT_LIMIT (amount={amount:,.2f} > {_settings.safety_max_amount:,.2f})")
                force = HUMAN_REVIEW
        except (ValueError, AttributeError):
            pass

        # Rail 2: New vendor.
        if not validation_result.get("vendor_known", False):
            vendor = field_map.get("VENDOR_NAME", "<unknown>")
            rails.append(f"RAIL_2_NEW_VENDOR (vendor='{vendor}')")
            force = force or HUMAN_REVIEW

        # Rail 3: Hard validation errors.
        error_issues = [i for i in issues if isinstance(i, dict) and i.get("severity") == "error"]
        if error_issues:
            rails.append(f"RAIL_3_VALIDATION_ERROR ({len(error_issues)} errors)")
            force = force or HUMAN_REVIEW

        # Rail 4: Very low confidence → reject (not auto or review).
        if confidence < _settings.human_review_threshold * 0.6:
            rails.append(f"RAIL_4_LOW_CONFIDENCE (conf={confidence:.3f})")
            force = REJECT

        return rails, force

    def _decide_with_llm(
        self,
        doc_type: str,
        final_confidence: float,
        is_valid: bool,
        vendor_known: bool,
        field_map: Dict[str, str],
        rails_triggered: List[str],
        validation_result: Dict[str, Any],
    ) -> str:
        """Ask the LLM for a routing decision; fall back to heuristics on failure.

        Safety rails have already been evaluated and were NOT triggered at this
        point, so the LLM is free to decide between auto_approve and human_review.
        The heuristic result is used if the LLM is unavailable or returns an
        invalid action.
        """
        # Compute heuristic action as guaranteed fallback.
        vendor_pattern = self.ltm.lookup_vendor_pattern(field_map.get("VENDOR_NAME", ""))
        heuristic_action = self._apply_routing_logic(
            final_confidence, is_valid, vendor_known, vendor_pattern
        )

        reasoner = get_reasoner()
        if not reasoner.is_available():
            self._think("LLM not available — using heuristic routing.")
            return heuristic_action

        self._think("LLM available — requesting LLM routing decision.")
        llm_output = reasoner.make_routing_decision(
            doc_type=doc_type,
            extraction_confidence=final_confidence,
            is_valid=is_valid,
            vendor_known=vendor_known,
            amount=field_map.get("TOTAL_AMOUNT", ""),
            safety_rails=rails_triggered,
            risk_level=validation_result.get("risk_level", "medium"),
            auto_threshold=_settings.auto_approve_threshold,
            review_threshold=_settings.human_review_threshold,
        )

        if llm_output and llm_output.get("action") in (AUTO_APPROVE, HUMAN_REVIEW, REJECT):
            llm_action = llm_output["action"]
            llm_reason = llm_output.get("reasoning", "")
            self._observe(
                f"LLM decided: {llm_action}. Reason: {llm_reason[:120]}"
            )
            return llm_action

        self._observe("LLM returned invalid action — falling back to heuristic routing.")
        return heuristic_action

    def _apply_routing_logic(
        self,
        confidence: float,
        is_valid: bool,
        vendor_known: bool,
        vendor_pattern: Optional[Dict[str, Any]],
    ) -> str:
        """Apply threshold-based routing (heuristic fallback)."""
        # Reject band.
        if not is_valid and confidence < _settings.human_review_threshold:
            return REJECT

        # Auto-approve band.
        if confidence >= _settings.auto_approve_threshold and is_valid and vendor_known:
            return AUTO_APPROVE

        # Borderline: confidence is high enough but context warrants review.
        if confidence >= _settings.auto_approve_threshold and not vendor_known:
            return HUMAN_REVIEW

        # Middle band.
        if _settings.human_review_threshold <= confidence < _settings.auto_approve_threshold:
            return HUMAN_REVIEW

        # Below review threshold.
        return REJECT

    # ── Output helpers ─────────────────────────────────────────────────────

    def _build_signals(
        self,
        confidence: float,
        validation_result: Dict[str, Any],
        field_map: Dict[str, str],
    ) -> Dict[str, Any]:
        return {
            "final_confidence": round(confidence, 4),
            "is_valid": validation_result.get("is_valid"),
            "vendor_known": validation_result.get("vendor_known"),
            "issue_count": len(validation_result.get("issues", [])),
            "amount": field_map.get("TOTAL_AMOUNT", ""),
            "vendor": field_map.get("VENDOR_NAME", ""),
            "auto_threshold": _settings.auto_approve_threshold,
            "review_threshold": _settings.human_review_threshold,
        }

    def _build_reasoning(
        self,
        action: str,
        confidence: float,
        validation_result: Dict[str, Any],
        rails_triggered: List[str],
        field_map: Dict[str, str],
        is_forced: bool,
        vendor_trust_boost: float = 0.0,
    ) -> str:
        vendor = field_map.get("VENDOR_NAME", "<not found>")
        amount = field_map.get("TOTAL_AMOUNT", "<not found>")
        is_valid = validation_result.get("is_valid", False)
        vendor_known = validation_result.get("vendor_known", False)

        if is_forced and rails_triggered:
            return (
                f"Safety rails FORCED action={action}. "
                f"Triggered rails: {'; '.join(rails_triggered)}. "
                f"Vendor='{vendor}', Amount={amount}, Confidence={confidence:.3f}."
            )

        trust_note = (
            f" Vendor trust boost applied ({vendor_trust_boost:+.3f})."
            if vendor_trust_boost != 0.0
            else ""
        )
        return (
            f"Routing decision: {action.upper()}. "
            f"Final confidence={confidence:.3f} "
            f"(auto≥{_settings.auto_approve_threshold}, review≥{_settings.human_review_threshold}). "
            f"Validation: is_valid={is_valid}, vendor_known={vendor_known}. "
            f"Vendor='{vendor}', Amount={amount}.{trust_note} "
            + (f"Safety rails: {'; '.join(rails_triggered)}." if rails_triggered else "No safety rails triggered.")
        )

    def _finalise(self, result: RouterResult) -> RouterResult:
        """Persist result to memory and return."""
        if self.memory:
            self.memory.set_context("router_result", {
                "action": result.action,
                "confidence": result.confidence,
                "reasoning": result.reasoning,
                "safety_rails_triggered": result.safety_rails_triggered,
            })
            self.memory.add_message("ai", result.reasoning, agent=self.AGENT_NAME)
        logger.info(
            "RouterAgent.run complete: action=%s confidence=%.3f rails=%s.",
            result.action, result.confidence, result.safety_rails_triggered,
        )
        return result

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
