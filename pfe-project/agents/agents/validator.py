"""Agent 3: Validator Agent.

Validates extracted fields against business rules and historical data:
1. Vendor existence check (PostgreSQL lookup).
2. Amount reasonableness check (compare to vendor's historical range).
3. Date validity check (not in future, not too old).
4. Anomaly detection (Isolation Forest if available, heuristics otherwise).
5. Redis-cached pattern check (has this vendor been flagged before?).

ReAct loop: Think → Query DB → Observe → Apply rules → Decide.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from agents.config import get_agent_settings
from agents.memory.long_term import LongTermMemory
from agents.memory.short_term import ShortTermMemory
from agents.tools.db_tools import lookup_vendor_history

logger = logging.getLogger(__name__)
_settings = get_agent_settings()


@dataclass
class ValidationIssue:
    """A single validation issue found during validation."""

    field_name: str
    issue_type: str  # "anomaly" | "rule_violation" | "missing" | "date_invalid"
    severity: str    # "warning" | "error"
    description: str


@dataclass
class ValidatorResult:
    """Output of the Validator Agent."""

    is_valid: bool
    issues: List[ValidationIssue]
    confidence_adjustment: float  # Can be positive (boost) or negative (penalty)
    reasoning: str
    vendor_known: bool = False
    amount_normal: bool = True
    date_valid: bool = True


class ValidatorAgent:
    """Validator Agent — checks extracted fields for sanity and business rule compliance.

    Queries the existing extraction_results database to understand what is
    "normal" for a given vendor, and applies configurable safety rules.

    Args:
        memory: Optional shared short-term memory.
        long_term_memory: Optional long-term memory for vendor pattern lookups.
    """

    AGENT_NAME = "validator"

    def __init__(
        self,
        memory: Optional[ShortTermMemory] = None,
        long_term_memory: Optional[LongTermMemory] = None,
    ) -> None:
        """Initialise the Validator Agent."""
        self.memory = memory
        self.ltm = long_term_memory or LongTermMemory()
        logger.info("ValidatorAgent initialised.")

    def run(
        self,
        fields: List[Dict[str, Any]],
        doc_type: str = "invoice",
    ) -> ValidatorResult:
        """Validate extracted fields using DB queries and business rules.

        ReAct steps:
        1. Think — which fields need validation and what rules apply?
        2. Act — lookup vendor in DB.
        3. Observe — is this vendor known? What's normal for them?
        4. Act — check amount vs historical range.
        5. Observe — is the amount anomalous?
        6. Act — validate date field.
        7. Observe — is the date in range?
        8. Decide — compile issues, compute confidence adjustment.

        Args:
            fields: List of field dicts from ExtractorAgent
                    (keys: field_name, value, confidence).
            doc_type: Document type for context.

        Returns:
            ValidatorResult with issues and confidence_adjustment.
        """
        logger.info("ValidatorAgent.run started: %d fields, doc_type=%s.", len(fields), doc_type)
        issues: List[ValidationIssue] = []
        confidence_adjustment = 0.0
        vendor_known = False
        amount_normal = True
        date_valid = True

        # Index fields by name for easy access.
        field_map: Dict[str, str] = {
            f.get("field_name", ""): str(f.get("value", ""))
            for f in fields
        }

        # ── Think ─────────────────────────────────────────────────────────
        self._think(
            f"I'll validate the {len(fields)} extracted fields for a {doc_type}. "
            "Checks: vendor exists in DB, amount is in normal range, "
            "date is valid, no anomalies."
        )

        # ── Vendor validation ─────────────────────────────────────────────
        vendor = field_map.get("VENDOR_NAME", "")
        if vendor:
            vendor_raw = lookup_vendor_history.invoke({"vendor_name": vendor})
            vendor_data = self._parse_json(vendor_raw, {})
            vendor_known = vendor_data.get("found", False)
            total_prior = vendor_data.get("total_invoices", 0)
            self._observe(
                f"Vendor '{vendor}': found={vendor_known}, total_invoices={total_prior}."
            )

            if not vendor_known:
                issues.append(ValidationIssue(
                    field_name="VENDOR_NAME",
                    issue_type="rule_violation",
                    severity="warning",
                    description=f"Vendor '{vendor}' not found in historical records. (Safety Rail 2: new vendor → human_review required)",
                ))
                confidence_adjustment -= 0.05
            else:
                confidence_adjustment += 0.02  # Known vendor → small boost.

            # Redis pattern check.
            if self.ltm.should_flag_vendor(vendor):
                issues.append(ValidationIssue(
                    field_name="VENDOR_NAME",
                    issue_type="anomaly",
                    severity="warning",
                    description=f"Vendor '{vendor}' flagged by long-term memory (high reject rate or unknown).",
                ))

            # Amount validation vs vendor history.
            amount_str = field_map.get("TOTAL_AMOUNT", "")
            amount = self._parse_amount(amount_str)
            if amount is not None and vendor_known and vendor_data.get("avg_amount", 0) > 0:
                avg = vendor_data["avg_amount"]
                min_amt = vendor_data.get("min_amount", 0)
                max_amt = vendor_data.get("max_amount", 1e9)

                if amount > max_amt * 2:
                    amount_normal = False
                    issues.append(ValidationIssue(
                        field_name="TOTAL_AMOUNT",
                        issue_type="anomaly",
                        severity="error",
                        description=(
                            f"Amount {amount:,.2f} is >2× the maximum ever seen for "
                            f"'{vendor}' (max={max_amt:,.2f}). Possible data error or fraud."
                        ),
                    ))
                    confidence_adjustment -= 0.10
                elif amount < min_amt * 0.1 and amount > 0:
                    amount_normal = False
                    issues.append(ValidationIssue(
                        field_name="TOTAL_AMOUNT",
                        issue_type="anomaly",
                        severity="warning",
                        description=(
                            f"Amount {amount:,.2f} is unusually small for '{vendor}' "
                            f"(min={min_amt:,.2f})."
                        ),
                    ))
                    confidence_adjustment -= 0.03
                else:
                    self._observe(
                        f"Amount {amount:,.2f} is within normal range "
                        f"[{min_amt:,.2f} – {max_amt:,.2f}] for '{vendor}'."
                    )
                    confidence_adjustment += 0.02

        # ── Safety Rail 1: Amount cap ──────────────────────────────────────
        amount_str = field_map.get("TOTAL_AMOUNT", "")
        amount = self._parse_amount(amount_str)
        if amount is not None and amount > _settings.safety_max_amount:
            issues.append(ValidationIssue(
                field_name="TOTAL_AMOUNT",
                issue_type="rule_violation",
                severity="error",
                description=(
                    f"Amount {amount:,.2f} exceeds the safety limit of "
                    f"{_settings.safety_max_amount:,.2f}. "
                    "Safety Rail 1: auto-approve is BLOCKED."
                ),
            ))
            confidence_adjustment -= 0.15

        # ── Date validation ────────────────────────────────────────────────
        date_str = field_map.get("INVOICE_DATE", "")
        if date_str:
            date_valid, date_issue = self._validate_date(date_str)
            if date_issue:
                issues.append(date_issue)
                confidence_adjustment -= 0.05
                self._observe(f"Date issue: {date_issue.description}")
            else:
                self._observe(f"Date '{date_str}' is valid.")

        # ── Missing critical fields ────────────────────────────────────────
        critical = ["INVOICE_ID", "TOTAL_AMOUNT", "VENDOR_NAME"]
        for fname in critical:
            if not field_map.get(fname, "").strip():
                issues.append(ValidationIssue(
                    field_name=fname,
                    issue_type="missing",
                    severity="warning",
                    description=f"Critical field '{fname}' is missing or empty.",
                ))
                confidence_adjustment -= 0.05

        # ── Decide ────────────────────────────────────────────────────────
        error_count = sum(1 for i in issues if i.severity == "error")
        warn_count = sum(1 for i in issues if i.severity == "warning")
        is_valid = error_count == 0

        reasoning = self._build_reasoning(
            is_valid, issues, confidence_adjustment, vendor_known, amount_normal, date_valid
        )

        result = ValidatorResult(
            is_valid=is_valid,
            issues=issues,
            confidence_adjustment=round(confidence_adjustment, 4),
            reasoning=reasoning,
            vendor_known=vendor_known,
            amount_normal=amount_normal,
            date_valid=date_valid,
        )

        if self.memory:
            self.memory.set_context("validator_result", {
                "is_valid": is_valid,
                "issue_count": len(issues),
                "error_count": error_count,
                "warning_count": warn_count,
                "confidence_adjustment": confidence_adjustment,
                "vendor_known": vendor_known,
                "reasoning": reasoning,
            })
            self.memory.add_message("ai", reasoning, agent=self.AGENT_NAME)

        logger.info(
            "ValidatorAgent.run complete: is_valid=%s, issues=%d (errors=%d, warnings=%d).",
            is_valid, len(issues), error_count, warn_count,
        )
        return result

    # ── Internal helpers ───────────────────────────────────────────────────

    def _validate_date(self, date_str: str) -> tuple:
        """Validate that a date string is in the past and not too old.

        Args:
            date_str: Raw date string from NER extraction.

        Returns:
            Tuple of (is_valid bool, ValidationIssue or None).
        """
        now = datetime.now(timezone.utc)
        formats = ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%B %d, %Y", "%b %d, %Y"]
        parsed_date = None
        for fmt in formats:
            try:
                parsed_date = datetime.strptime(date_str.strip(), fmt).replace(tzinfo=timezone.utc)
                break
            except ValueError:
                continue

        if parsed_date is None:
            return False, ValidationIssue(
                field_name="INVOICE_DATE",
                issue_type="date_invalid",
                severity="warning",
                description=f"Could not parse date '{date_str}' with known formats.",
            )

        # Date must not be in the future.
        if parsed_date > now:
            return False, ValidationIssue(
                field_name="INVOICE_DATE",
                issue_type="date_invalid",
                severity="error",
                description=f"Invoice date '{date_str}' is in the future.",
            )

        # Date must not be older than 5 years.
        age_years = (now - parsed_date).days / 365
        if age_years > 5:
            return False, ValidationIssue(
                field_name="INVOICE_DATE",
                issue_type="date_invalid",
                severity="warning",
                description=f"Invoice date '{date_str}' is more than 5 years old ({age_years:.1f} years).",
            )

        return True, None

    @staticmethod
    def _parse_amount(amount_str: str) -> Optional[float]:
        """Parse a currency amount string to float, returning None on failure."""
        if not amount_str:
            return None
        # Strip currency symbols and commas.
        cleaned = re.sub(r"[^\d.]", "", amount_str.replace(",", ""))
        try:
            return float(cleaned)
        except ValueError:
            return None

    def _build_reasoning(
        self,
        is_valid: bool,
        issues: List[ValidationIssue],
        confidence_adjustment: float,
        vendor_known: bool,
        amount_normal: bool,
        date_valid: bool,
    ) -> str:
        issue_summary = (
            "; ".join(f"[{i.severity.upper()}] {i.description}" for i in issues)
            if issues
            else "No issues found."
        )
        adj_sign = "+" if confidence_adjustment >= 0 else ""
        return (
            f"Validation {'PASSED' if is_valid else 'FAILED'}. "
            f"vendor_known={vendor_known}, amount_normal={amount_normal}, date_valid={date_valid}. "
            f"Issues: {issue_summary}. "
            f"Confidence adjustment: {adj_sign}{confidence_adjustment:.3f}."
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
