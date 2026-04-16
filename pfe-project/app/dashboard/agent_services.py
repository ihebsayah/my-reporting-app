"""Agent dashboard data service — fetches data from the agent microservice.

This module is an extension to the existing ``DashboardDataService``.
It calls the agent service REST API (not importing agents code directly)
so it works even when the agent service is running in a separate container.

If the agent service is unreachable, every method returns a safe default
so the dashboard never crashes.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

_AGENT_URL: str = os.environ.get("AGENT_SERVICE_URL", "").rstrip("/")
_TIMEOUT: float = float(os.environ.get("AGENT_SERVICE_TIMEOUT", "5.0"))


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class AgentStatus:
    """Agent service health and rollback state."""

    reachable: bool
    agents_enabled: bool
    rolled_back: bool
    last_rollback_reason: Optional[str]
    version: str


@dataclass
class AgentAccuracyStats:
    """Sliding-window accuracy from the agent monitoring endpoint."""

    window: int
    total_decisions: int
    accuracy: float
    override_rate: float
    rollback_needed: bool
    rollback_reasons: List[str]
    accuracy_threshold: float
    override_rate_threshold: float


@dataclass
class AgentDecisionLog:
    """One agent decision record for the dashboard."""

    document_id: str
    action: str
    confidence: float
    doc_type: str
    agents_used: List[str]
    safety_rails_triggered: List[str]
    fallback_used: bool
    duration_ms: int
    reasoning: str
    validation_issues: List[Dict[str, Any]] = field(default_factory=list)


# ── Service ───────────────────────────────────────────────────────────────────


class AgentDataService:
    """Fetches monitoring data from the agent microservice for the dashboard.

    All methods return safe defaults when the agent service is unreachable.
    """

    def __init__(self, agent_url: Optional[str] = None) -> None:
        self.agent_url = (agent_url or _AGENT_URL).rstrip("/")
        self.available = bool(self.agent_url)

    # ── Status ────────────────────────────────────────────────────────────────

    def get_status(self) -> AgentStatus:
        """Fetch agent service health and rollback state.

        Returns:
            AgentStatus — all fields default to safe values on failure.
        """
        if not self.available:
            return AgentStatus(
                reachable=False, agents_enabled=False, rolled_back=False,
                last_rollback_reason="AGENT_SERVICE_URL not configured",
                version="N/A",
            )
        try:
            resp = httpx.get(f"{self.agent_url}/agents/status", timeout=_TIMEOUT)
            resp.raise_for_status()
            d = resp.json()
            return AgentStatus(
                reachable=True,
                agents_enabled=d.get("agents_enabled", False),
                rolled_back=d.get("rolled_back", False),
                last_rollback_reason=d.get("last_rollback_reason"),
                version=d.get("agent_service_version", "unknown"),
            )
        except Exception as exc:
            logger.warning("AgentDataService.get_status failed: %s", exc)
            return AgentStatus(
                reachable=False, agents_enabled=False, rolled_back=False,
                last_rollback_reason=str(exc), version="unreachable",
            )

    # ── Accuracy ──────────────────────────────────────────────────────────────

    def get_accuracy(self) -> Optional[AgentAccuracyStats]:
        """Fetch accuracy stats from the monitoring endpoint.

        Returns:
            AgentAccuracyStats or None if the service is unavailable.
        """
        if not self.available:
            return None
        try:
            resp = httpx.get(f"{self.agent_url}/agents/monitoring/accuracy", timeout=_TIMEOUT)
            resp.raise_for_status()
            d = resp.json()
            return AgentAccuracyStats(
                window=d.get("window", 100),
                total_decisions=d.get("total_decisions", 0),
                accuracy=d.get("accuracy", 0.0),
                override_rate=d.get("override_rate", 0.0),
                rollback_needed=d.get("rollback_needed", False),
                rollback_reasons=d.get("rollback_reasons", []),
                accuracy_threshold=d.get("accuracy_threshold", 0.85),
                override_rate_threshold=d.get("override_rate_threshold", 0.30),
            )
        except Exception as exc:
            logger.warning("AgentDataService.get_accuracy failed: %s", exc)
            return None

    # ── Live extraction (try-document) ────────────────────────────────────────

    def run_agent_extraction(
        self,
        text: str,
        document_id: Optional[str] = None,
    ) -> Optional[AgentDecisionLog]:
        """Send a document directly to the agent service and return its decision.

        Used by the "Try Document" panel in the dashboard.

        Args:
            text: Raw document text.
            document_id: Optional ID tag.

        Returns:
            AgentDecisionLog or None on failure.
        """
        if not self.available:
            return None
        try:
            resp = httpx.post(
                f"{self.agent_url}/agents/extract",
                json={"text": text, "document_id": document_id, "metadata": {"source": "dashboard"}},
                timeout=30.0,
            )
            resp.raise_for_status()
            d = resp.json()
            return AgentDecisionLog(
                document_id=d.get("document_id", ""),
                action=d.get("action", "unknown"),
                confidence=float(d.get("confidence", 0.0)),
                doc_type=d.get("doc_type", "unknown"),
                agents_used=d.get("agents_used", []),
                safety_rails_triggered=d.get("safety_rails_triggered", []),
                fallback_used=bool(d.get("fallback_used", False)),
                duration_ms=int(d.get("duration_ms", 0)),
                reasoning=d.get("agent_reasoning", ""),
                validation_issues=d.get("validation_issues", []),
            )
        except Exception as exc:
            logger.warning("AgentDataService.run_agent_extraction failed: %s", exc)
            return None

    # ── Feedback ──────────────────────────────────────────────────────────────

    def submit_feedback(
        self,
        document_id: str,
        agent_decision: str,
        human_outcome: str,
        vendor: str = "",
        amount: float = 0.0,
        notes: str = "",
    ) -> bool:
        """Submit human feedback to the agent service for real-time learning.

        Args:
            document_id: The processed document ID.
            agent_decision: What the agent decided.
            human_outcome: Human's final outcome (approve/reject/review).
            vendor: Extracted vendor name.
            amount: Extracted invoice amount.
            notes: Optional human notes.

        Returns:
            True if the feedback was recorded, False otherwise.
        """
        if not self.available:
            return False
        try:
            resp = httpx.post(
                f"{self.agent_url}/agents/feedback",
                json={
                    "document_id": document_id,
                    "agent_decision": agent_decision,
                    "human_outcome": human_outcome,
                    "vendor": vendor,
                    "amount": amount,
                    "notes": notes,
                },
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json().get("recorded", False)
        except Exception as exc:
            logger.warning("AgentDataService.submit_feedback failed: %s", exc)
            return False

    # ── Admin ─────────────────────────────────────────────────────────────────

    def disable_agents(self, reason: str = "dashboard_manual") -> bool:
        """Disable agents via the admin endpoint."""
        try:
            resp = httpx.post(
                f"{self.agent_url}/agents/admin/disable",
                params={"reason": reason},
                timeout=_TIMEOUT,
            )
            return resp.status_code == 200
        except Exception:
            return False

    def enable_agents(self) -> bool:
        """Re-enable agents via the admin endpoint."""
        try:
            resp = httpx.post(f"{self.agent_url}/agents/admin/enable", timeout=_TIMEOUT)
            return resp.status_code == 200
        except Exception:
            return False
