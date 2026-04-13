"""Agent bridge — thin adapter that calls the agent microservice from FastAPI.

This module is the ONLY point of contact between the existing FastAPI app
(app/) and the new agent service (agents/).  It:

1. POSTs to ``http://<AGENT_SERVICE_URL>/agents/extract`` via httpx.
2. Returns a populated ``AgentDecisionResponse`` on success.
3. Returns ``None`` silently on any failure (so the existing pipeline
   continues unaffected — agents are additive, never blocking).

By keeping this in a separate module we guarantee:
- No agent code is imported into the existing ``app/`` package.
- If the agent service is down, ``None`` is returned and the existing
  ``PipelineResponse`` is returned to the caller unchanged.
- The ``AGENT_SERVICE_URL`` env var fully controls whether agents are used.

Usage in routes.py::

    from app.api.agent_bridge import call_agent_service

    agent_result = call_agent_service(text=request.text, document_id=doc_id)
    # agent_result is AgentDecisionResponse | None
"""

import logging
import os
from typing import Optional

import httpx

from app.api.schemas import AgentDecisionResponse, AgentValidationIssue

logger = logging.getLogger(__name__)

# Read from env — default keeps agents disabled unless explicitly configured.
_AGENT_SERVICE_URL: str = os.environ.get("AGENT_SERVICE_URL", "").rstrip("/")
_AGENT_TIMEOUT: float = float(os.environ.get("AGENT_SERVICE_TIMEOUT", "10.0"))


def agents_enabled() -> bool:
    """Return True if the agent service URL is configured in the environment.

    Returns:
        True if ``AGENT_SERVICE_URL`` is set and non-empty.
    """
    return bool(_AGENT_SERVICE_URL)


def call_agent_service(
    text: str,
    document_id: Optional[str] = None,
) -> Optional[AgentDecisionResponse]:
    """Call the agent service and return a structured decision, or None on failure.

    This is a **synchronous** call kept intentionally simple so it can be
    dropped into the existing synchronous FastAPI route without async changes.
    The timeout is short (default 10s) so a slow or unresponsive agent service
    never blocks the existing pipeline.

    Args:
        text: Raw document text (same text passed to the ML pipeline).
        document_id: Optional document ID for traceability.

    Returns:
        ``AgentDecisionResponse`` on success, or ``None`` if agents are
        disabled, the service is unreachable, or any error occurs.
    """
    if not agents_enabled():
        logger.debug("Agent service not configured (AGENT_SERVICE_URL unset) — skipping.")
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

        return AgentDecisionResponse(
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
