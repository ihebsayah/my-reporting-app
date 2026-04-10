"""Safety Rails Enforcer — validates agent decisions against hard business rules.

This class is a stateless guard that can be called *after* an agent produces
a decision to verify that no safety rail has been silently violated.  It is
separate from the per-agent rail checks so it can be used as an independent
audit layer at the API boundary.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agents.config import get_agent_settings

logger = logging.getLogger(__name__)
_settings = get_agent_settings()


@dataclass
class SafetyCheckResult:
    """Result of a safety rails audit."""

    passed: bool
    violated_rails: List[str] = field(default_factory=list)
    corrected_action: Optional[str] = None  # Override if a rail was violated
    notes: str = ""


class SafetyRailsEnforcer:
    """Independent safety-rail audit layer.

    Verifies that a proposed agent action does not violate any hard rules.
    If violations are found the enforcer overrides the action.

    Rails enforced:
    1. Amount > SAFETY_MAX_AMOUNT → action must be ``human_review`` (not auto_approve).
    2. New vendor (not in DB) → action must be ``human_review``.
    3. Anomaly flagged by Validator → action must NOT be ``auto_approve``.
    4. Override Integrity → if human already decided, record discrepancy.

    Usage::

        enforcer = SafetyRailsEnforcer()
        result = enforcer.check(
            proposed_action="auto_approve",
            confidence=0.92,
            fields={"TOTAL_AMOUNT": "150000", "VENDOR_NAME": "Acme"},
            validation_result={"is_valid": True, "vendor_known": False},
        )
        if not result.passed:
            final_action = result.corrected_action
    """

    def check(
        self,
        proposed_action: str,
        confidence: float,
        fields: Dict[str, str],
        validation_result: Dict[str, Any],
        human_decision: Optional[str] = None,
    ) -> SafetyCheckResult:
        """Audit the proposed action against all safety rails.

        Args:
            proposed_action: The action returned by RouterAgent.
            confidence: Final confidence score used for routing.
            fields: Flat dict of field_name → value.
            validation_result: Output dict from ValidatorAgent.
            human_decision: Optional human decision (for override integrity check).

        Returns:
            SafetyCheckResult — may contain a corrected_action override.
        """
        violations: List[str] = []
        corrected_action = proposed_action

        # Rail 1: Amount limit.
        amount = self._parse_amount(fields.get("TOTAL_AMOUNT", ""))
        if amount is not None and amount > _settings.safety_max_amount:
            if proposed_action == "auto_approve":
                violations.append(
                    f"RAIL_1: Amount {amount:,.2f} > {_settings.safety_max_amount:,.2f}. "
                    "auto_approve is BLOCKED."
                )
                corrected_action = "human_review"
                logger.warning(
                    "Safety Rail 1 VIOLATED: amount=%.2f exceeds limit. "
                    "Overriding %s → human_review.",
                    amount, proposed_action,
                )

        # Rail 2: New vendor.
        vendor_known = validation_result.get("vendor_known", False)
        if not vendor_known and proposed_action == "auto_approve":
            vendor = fields.get("VENDOR_NAME", "<unknown>")
            violations.append(
                f"RAIL_2: Vendor '{vendor}' not in DB. auto_approve BLOCKED."
            )
            corrected_action = "human_review"
            logger.warning("Safety Rail 2 VIOLATED: unknown vendor '%s'. Overriding → human_review.", vendor)

        # Rail 3: Anomaly suppression check.
        issues = validation_result.get("issues", [])
        error_issues = [
            i for i in issues
            if isinstance(i, dict) and i.get("severity") == "error"
        ]
        if error_issues and proposed_action == "auto_approve":
            violations.append(
                f"RAIL_3: Validation errors present ({len(error_issues)} errors). "
                "auto_approve is BLOCKED."
            )
            corrected_action = "human_review"
            logger.warning("Safety Rail 3 VIOLATED: validation errors present. Overriding → human_review.")

        # Rail 4: Override integrity — record discrepancy, don't change human decision.
        if human_decision and human_decision != proposed_action:
            violations.append(
                f"RAIL_4: Human decision '{human_decision}' differs from agent decision "
                f"'{proposed_action}'. Discrepancy recorded."
            )
            logger.info(
                "Safety Rail 4 (override integrity): human=%s agent=%s",
                human_decision, proposed_action,
            )
            # Do NOT change corrected_action — human wins, we just record.

        passed = len([v for v in violations if "RAIL_1" in v or "RAIL_2" in v or "RAIL_3" in v]) == 0
        notes = "; ".join(violations) if violations else "All safety rails passed."

        if not passed:
            logger.info(
                "SafetyRailsEnforcer: %d violation(s), corrected_action=%s.",
                len(violations), corrected_action,
            )

        return SafetyCheckResult(
            passed=passed,
            violated_rails=violations,
            corrected_action=corrected_action if not passed else None,
            notes=notes,
        )

    @staticmethod
    def _parse_amount(amount_str: str) -> Optional[float]:
        """Parse a currency string to float."""
        if not amount_str:
            return None
        cleaned = re.sub(r"[^\d.]", "", amount_str.replace(",", ""))
        try:
            return float(cleaned)
        except ValueError:
            return None
