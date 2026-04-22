"""Agent bridge — thin adapter that calls the agent microservice from FastAPI.

This module is the ONLY point of contact between the existing FastAPI app
(app/) and the new agent service (agents/).  It:

1. Implements **Canary routing**: only ``CANARY_RATE`` fraction (default 5%)
   of requests are forwarded to the agent service. The rest return ``None``
   immediately, leaving the existing pipeline response unchanged.
2. POSTs to ``http://<AGENT_SERVICE_URL>/agents/extract`` via httpx.
3. Logs every canary call to ``artifacts/canary/canary.jsonl`` for offline
   agreement analysis via ``scripts/canary_report.py``.
4. Returns ``None`` silently on any failure — agents are always additive.

Environment variables:
    AGENT_SERVICE_URL    URL of the agent microservice (e.g. http://localhost:8001).
                         If unset/empty, all agent calls are skipped.
    CANARY_RATE          Fraction of requests forwarded to agents (default 0.05 = 5%).
    AGENT_SERVICE_TIMEOUT  Per-request timeout in seconds (default 10.0).
    CANARY_LOG_DIR       Directory for canary JSONL logs (default artifacts/canary).

Usage in routes.py::

    from app.api.agent_bridge import call_agent_service

    agent_result = call_agent_service(
        text=request.text,
        document_id=doc_id,
        main_decision=result.overall_decision,  # for canary comparison
    )
"""

import json
import logging
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx

from app.api.schemas import AgentDecisionResponse, AgentValidationIssue

logger = logging.getLogger(__name__)

# ── Environment config ────────────────────────────────────────────────────────
_AGENT_SERVICE_URL: str = os.environ.get("AGENT_SERVICE_URL", "").rstrip("/")
_AGENT_TIMEOUT: float    = float(os.environ.get("AGENT_SERVICE_TIMEOUT", "10.0"))
_CANARY_RATE: float      = float(os.environ.get("CANARY_RATE", "0.05"))
_CANARY_LOG_DIR: Path    = Path(os.environ.get("CANARY_LOG_DIR", "artifacts/canary"))

# Ensure log directory exists at import time (fail silently).
try:
    _CANARY_LOG_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass


def _write_canary_log(
    document_id: str,
    main_decision: str,
    agent_action: str,
    agent_confidence: float,
    agent_rails: list,
    duration_ms: int,
    fallback_used: bool,
) -> None:
    """Append one canary comparison record to the rolling JSONL log."""
    # Map agent action → main vocabulary for comparison
    _action_map = {"auto_approve": "auto", "human_review": "review", "reject": "reject"}
    agent_mapped = _action_map.get(agent_action, agent_action)
    agreement = "agree" if main_decision == agent_mapped else "disagree"

    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "document_id": document_id,
        "main_decision": main_decision,
        "agent_action": agent_action,
        "agent_action_mapped": agent_mapped,
        "agreement": agreement,
        "agent_confidence": agent_confidence,
        "rails_triggered": agent_rails,
        "duration_ms": duration_ms,
        "fallback_used": fallback_used,
    }
    log_file = _CANARY_LOG_DIR / "canary.jsonl"
    try:
        with open(log_file, "a") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception as exc:
        logger.debug("canary log write failed: %s", exc)


def agents_enabled() -> bool:
    """Return True if the agent service URL is configured in the environment."""
    return bool(_AGENT_SERVICE_URL)


def canary_rate() -> float:
    """Return the configured canary sampling rate (0.0–1.0)."""
    return _CANARY_RATE


def call_agent_service(
    text: str,
    document_id: Optional[str] = None,
    main_decision: Optional[str] = None,
) -> Optional[AgentDecisionResponse]:
    """Call the agent service and return a structured decision, or None on failure.

    Implements **Canary routing**: only ``CANARY_RATE`` fraction of requests
    are forwarded. On success, logs the main vs agent comparison to
    ``artifacts/canary/canary.jsonl`` for offline analysis.

    Args:
        text: Raw document text (same text passed to the ML pipeline).
        document_id: Optional document ID for traceability.
        main_decision: The main pipeline decision ("auto"/"review"/"reject");
            used for canary agreement tracking.

    Returns:
        ``AgentDecisionResponse`` on success, or ``None`` if agents are
        disabled, outside canary sample, or any error occurs.
    """
    if not agents_enabled():
        logger.debug("Agent service not configured (AGENT_SERVICE_URL unset) — skipping.")
        return None

    # ── Canary gate: probabilistic sampling ───────────────────────────────
    if random.random() > _CANARY_RATE:
        logger.debug(
            "agent_bridge: canary gate — doc_id=%s skipped (rate=%.0f%%).",
            document_id, _CANARY_RATE * 100,
        )
        return None

    url = f"{_AGENT_SERVICE_URL}/agents/extract"
    payload = {
        "text": text,
        "document_id": document_id,
        "metadata": {"source": "pipeline_bridge"},
    }

    try:
        logger.info(
            "agent_bridge: calling agent service at %s (doc_id=%s).", url, document_id
        )
        response = httpx.post(url, json=payload, timeout=_AGENT_TIMEOUT)
        response.raise_for_status()
        data = response.json()

        issues = [
            AgentValidationIssue(
                field_name=i.get("field_name", ""),
                issue_type=i.get("issue_type", ""),
                severity=i.get("severity", "warning"),
                description=i.get("description", ""),
            )
            for i in data.get("validation_issues", [])
        ]

        decision = AgentDecisionResponse(
            action=data.get("action", "human_review"),
            confidence=float(data.get("confidence", 0.0)),
            agent_reasoning=data.get("agent_reasoning", ""),
            doc_type=data.get("doc_type", "unknown"),
            session_id=data.get("session_id", ""),
            duration_ms=int(data.get("duration_ms", 0)),
            agents_used=data.get("agents_used", []),
            safety_rails_triggered=data.get("safety_rails_triggered", []),
            validation_issues=issues,
            fallback_used=bool(data.get("fallback_used", False)),
            agent_service_version=data.get("agent_service_version", "unknown"),
        )

        # ── Canary log ────────────────────────────────────────────────────
        _write_canary_log(
            document_id=document_id or "",
            main_decision=main_decision or "unknown",
            agent_action=decision.action,
            agent_confidence=decision.confidence,
            agent_rails=decision.safety_rails_triggered,
            duration_ms=decision.duration_ms,
            fallback_used=decision.fallback_used,
        )
        logger.info(
            "agent_bridge: canary call — doc_id=%s main=%s agent=%s conf=%.2f agree=%s",
            document_id, main_decision, decision.action, decision.confidence,
            main_decision == {"auto_approve": "auto", "human_review": "review",
                              "reject": "reject"}.get(decision.action, decision.action),
        )
        return decision

    except httpx.TimeoutException:
        logger.warning(
            "agent_bridge: timeout calling agent service at %s (%.1fs limit). "
            "Continuing without agent enrichment.",
            url, _AGENT_TIMEOUT,
        )
    except httpx.HTTPStatusError as exc:
        logger.warning(
            "agent_bridge: HTTP %d from agent service: %s. Continuing without agent.",
            exc.response.status_code, exc,
        )
    except Exception as exc:
        logger.warning(
            "agent_bridge: unexpected error calling agent service: %s. "
            "Continuing without agent enrichment.", exc,
        )

    return None
