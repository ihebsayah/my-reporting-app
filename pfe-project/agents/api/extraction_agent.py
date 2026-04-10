"""FastAPI router for the Agent Service — POST /agents/extract and related endpoints.

This router is mounted by ``agents/main.py`` as a separate microservice running
on port 8001.  The existing FastAPI app on port 8000 is NOT modified.

Endpoints
---------
POST /agents/extract
    Process a document through all 5 agents. Returns decision + reasoning.

POST /agents/feedback
    Record human feedback for real-time learning.

GET  /agents/status
    Health check + current agent enable/disable state.

GET  /agents/monitoring/accuracy
    Accuracy stats and rollback recommendation.

POST /agents/admin/disable
    Manually disable all agents (runtime, no restart needed).

POST /agents/admin/enable
    Re-enable agents after an investigation.

GET  /agents/monitoring/rollback-status
    Current auto-rollback state.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from agents.config import get_agent_settings
from agents.master_agent import AgentExtractionResult, MasterAgent
from agents.monitoring.accuracy_tracker import AccuracyTracker
from agents.monitoring.agent_logger import AgentLogger
from agents.monitoring.auto_rollback import AutoRollbackMonitor
from agents.monitoring.safety_rails import SafetyRailsEnforcer

logger = logging.getLogger(__name__)
_settings = get_agent_settings()

# ── Singletons ─────────────────────────────────────────────────────────────────
# These are created once at module load and reused across requests.
_master_agent = MasterAgent()
_safety_enforcer = SafetyRailsEnforcer()
_accuracy_tracker = AccuracyTracker()
_rollback_monitor = AutoRollbackMonitor()
_agent_logger = AgentLogger()

router = APIRouter(prefix="/agents", tags=["agents"])


# ── Request / Response schemas ─────────────────────────────────────────────────


class AgentExtractRequest(BaseModel):
    """Request body for POST /agents/extract."""

    text: str = Field(..., description="Raw document text to process.", min_length=1)
    document_id: Optional[str] = Field(
        default=None,
        description="Optional document ID. Auto-generated from text prefix if not provided.",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata from the calling FastAPI pipeline.",
    )


class ValidationIssueResponse(BaseModel):
    """One validation issue."""

    field_name: str
    issue_type: str
    severity: str
    description: str


class ExtractedFieldResponse(BaseModel):
    """One extracted field from the agent pipeline."""

    field_name: str
    value: str
    confidence: float
    sources: List[str] = Field(default_factory=list)
    decision: str = "review"


class AgentExtractResponse(BaseModel):
    """Response body for POST /agents/extract."""

    document_id: str
    action: str = Field(..., description="auto_approve | human_review | reject")
    confidence: float
    agent_reasoning: str
    doc_type: str
    extracted_fields: List[ExtractedFieldResponse]
    validation_issues: List[ValidationIssueResponse]
    safety_rails_triggered: List[str]
    session_id: str
    duration_ms: int
    agents_used: List[str]
    fallback_used: bool
    processed_at: str
    agent_service_version: str


class FeedbackRequest(BaseModel):
    """Request body for POST /agents/feedback."""

    document_id: str
    agent_decision: str = Field(..., description="What the agent originally decided.")
    human_outcome: str = Field(..., description="Human's final decision: approve | reject | review")
    vendor: Optional[str] = Field(default="")
    amount: Optional[float] = Field(default=0.0)
    notes: Optional[str] = Field(default=None)


class FeedbackResponse(BaseModel):
    """Response body for POST /agents/feedback."""

    document_id: str
    recorded: bool
    message: str
    learned: bool


class AgentStatusResponse(BaseModel):
    """Response body for GET /agents/status."""

    agents_enabled: bool
    rolled_back: bool
    last_rollback_reason: Optional[str]
    agent_service_version: str
    checked_at: str


class AccuracyResponse(BaseModel):
    """Response body for GET /agents/monitoring/accuracy."""

    window: int
    total_decisions: int
    accuracy: float
    override_rate: float
    rollback_needed: bool
    rollback_reasons: List[str]
    accuracy_threshold: float
    override_rate_threshold: float


# ── Endpoints ──────────────────────────────────────────────────────────────────


@router.post("/extract", response_model=AgentExtractResponse)
def agent_extract(
    request: AgentExtractRequest,
    background_tasks: BackgroundTasks,
) -> AgentExtractResponse:
    """Process a document through all 5 AI agents.

    The Master Agent orchestrates:
    1. ClassifierAgent  — document type
    2. ExtractorAgent   — field extraction
    3. ValidatorAgent   — business rule checks
    4. RouterAgent      — routing decision (auto_approve / human_review / reject)

    Safety rails are enforced at two layers:
    - Within RouterAgent (per-agent check).
    - By SafetyRailsEnforcer after the agent pipeline (API boundary audit).

    If agents are globally disabled or crash, falls back to ``human_review``
    so no document is ever lost.

    Args:
        request: Document text and optional metadata.
        background_tasks: FastAPI background task runner (used for JSONL logging).

    Returns:
        AgentExtractResponse with decision, reasoning, and all field data.

    Raises:
        HTTPException 503: If agents are disabled and no fallback is possible.
    """
    # Runtime enable/disable check (includes auto-rollback state).
    if not AutoRollbackMonitor.agents_enabled():
        logger.warning("agent_extract called while agents are disabled — returning safe fallback.")
        now = datetime.now(timezone.utc).isoformat()
        return AgentExtractResponse(
            document_id=request.document_id or "disabled",
            action="human_review",
            confidence=0.0,
            agent_reasoning=(
                "⚠️ AI agents are currently DISABLED (auto-rollback or manual disable). "
                "Routing to human review as a safe default. "
                "Use POST /agents/admin/enable to re-activate."
            ),
            doc_type="unknown",
            extracted_fields=[],
            validation_issues=[],
            safety_rails_triggered=["AGENTS_GLOBALLY_DISABLED"],
            session_id="disabled",
            duration_ms=0,
            agents_used=[],
            fallback_used=True,
            processed_at=now,
            agent_service_version=_settings.agent_service_version,
        )

    doc_id = request.document_id or _auto_document_id(request.text)
    logger.info("agent_extract: document_id=%s text_len=%d.", doc_id, len(request.text))

    # Run the full agent pipeline.
    result: AgentExtractionResult = _master_agent.process_document(
        document_id=doc_id,
        text=request.text,
        metadata=request.metadata,
    )

    # ── Safety rails audit (API boundary) ─────────────────────────────────
    field_map: Dict[str, str] = {
        f.get("field_name", ""): str(f.get("value", ""))
        for f in result.extracted_fields
    }
    # Build a minimal validation_result dict for the enforcer.
    validation_summary = {
        "is_valid": len([i for i in result.validation_issues if i.get("severity") == "error"]) == 0,
        "vendor_known": not any(
            "RAIL_2_NEW_VENDOR" in r for r in result.safety_rails_triggered
        ),
        "issues": result.validation_issues,
    }
    rail_check = _safety_enforcer.check(
        proposed_action=result.action,
        confidence=result.confidence,
        fields=field_map,
        validation_result=validation_summary,
    )
    if not rail_check.passed and rail_check.corrected_action:
        logger.warning(
            "SafetyRailsEnforcer overrode action %s → %s. Violations: %s",
            result.action,
            rail_check.corrected_action,
            rail_check.violated_rails,
        )
        result.action = rail_check.corrected_action
        result.safety_rails_triggered.extend(
            [v for v in rail_check.violated_rails if v not in result.safety_rails_triggered]
        )

    now = datetime.now(timezone.utc).isoformat()

    # Log asynchronously so it doesn't block the response.
    background_tasks.add_task(
        _agent_logger.log_decision,
        document_id=doc_id,
        session_id=result.session_id,
        final_decision=result.action,
        confidence=result.confidence,
        doc_type=result.doc_type,
        agents_used=result.agents_used,
        duration_ms=result.duration_ms,
        extracted_fields=result.extracted_fields,
        validation_issues=result.validation_issues,
        safety_rails_triggered=result.safety_rails_triggered,
        fallback_used=result.fallback_used,
        error=result.error,
    )

    # Schedule periodic accuracy check in background.
    background_tasks.add_task(_check_accuracy_in_background)

    return AgentExtractResponse(
        document_id=result.document_id,
        action=result.action,
        confidence=result.confidence,
        agent_reasoning=result.agent_reasoning,
        doc_type=result.doc_type,
        extracted_fields=[
            ExtractedFieldResponse(
                field_name=f.get("field_name", ""),
                value=str(f.get("value", "")),
                confidence=float(f.get("confidence", 0.0)),
                sources=list(f.get("sources", [])),
                decision=f.get("decision", "review"),
            )
            for f in result.extracted_fields
        ],
        validation_issues=[
            ValidationIssueResponse(
                field_name=i.get("field_name", ""),
                issue_type=i.get("issue_type", ""),
                severity=i.get("severity", "warning"),
                description=i.get("description", ""),
            )
            for i in result.validation_issues
        ],
        safety_rails_triggered=result.safety_rails_triggered,
        session_id=result.session_id,
        duration_ms=result.duration_ms,
        agents_used=result.agents_used,
        fallback_used=result.fallback_used,
        processed_at=now,
        agent_service_version=_settings.agent_service_version,
    )


@router.post("/feedback", response_model=FeedbackResponse)
def record_feedback(request: FeedbackRequest) -> FeedbackResponse:
    """Record human feedback after a human approves / rejects / overrides an agent decision.

    This triggers REAL-TIME learning:
    - Redis patterns are updated immediately.
    - PostgreSQL agent_patterns table is updated for durability.
    - The agent logger records the feedback event.

    Args:
        request: Feedback payload with human and agent decisions.

    Returns:
        FeedbackResponse confirming the recording.
    """
    logger.info(
        "record_feedback: document_id=%s human=%s agent=%s.",
        request.document_id, request.human_outcome, request.agent_decision,
    )
    try:
        _master_agent.record_human_feedback(
            document_id=request.document_id,
            human_outcome=request.human_outcome,
            agent_decision=request.agent_decision,
            vendor=request.vendor or "",
            amount=request.amount or 0.0,
            notes=request.notes,
        )
        _agent_logger.log_feedback(
            document_id=request.document_id,
            agent_decision=request.agent_decision,
            human_outcome=request.human_outcome,
            vendor=request.vendor or "",
            amount=request.amount or 0.0,
            notes=request.notes,
        )
        return FeedbackResponse(
            document_id=request.document_id,
            recorded=True,
            message="Feedback recorded and learning signals updated.",
            learned=True,
        )
    except Exception as exc:
        logger.error("record_feedback failed: %s.", exc)
        return FeedbackResponse(
            document_id=request.document_id,
            recorded=False,
            message=f"Failed to record feedback: {exc}",
            learned=False,
        )


@router.get("/status", response_model=AgentStatusResponse)
def agent_status() -> AgentStatusResponse:
    """Return current agent health and enable/disable status.

    Returns:
        AgentStatusResponse with rollback state and version.
    """
    rollback_status = AutoRollbackMonitor.status()
    return AgentStatusResponse(
        agents_enabled=rollback_status["agents_enabled"],
        rolled_back=rollback_status["rolled_back"],
        last_rollback_reason=rollback_status["last_rollback_reason"],
        agent_service_version=_settings.agent_service_version,
        checked_at=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/monitoring/accuracy", response_model=AccuracyResponse)
def get_accuracy() -> AccuracyResponse:
    """Return accuracy statistics and rollback recommendation for the monitoring dashboard.

    Returns:
        AccuracyResponse with accuracy %, override rate, and rollback flag.
    """
    summary = _accuracy_tracker.summary_dict()
    return AccuracyResponse(**summary)


@router.get("/monitoring/rollback-status")
def get_rollback_status() -> dict:
    """Return the current auto-rollback state.

    Returns:
        Dictionary with rollback status and last trigger reason.
    """
    return AutoRollbackMonitor.status()


@router.post("/admin/disable")
def disable_agents(reason: str = "manual_admin") -> dict:
    """Manually disable all agents at runtime (no service restart needed).

    Args:
        reason: Human-readable reason for disabling (logged).

    Returns:
        Confirmation dictionary.
    """
    AutoRollbackMonitor.disable_agents(reason=reason)
    logger.warning("Agent service MANUALLY DISABLED via admin endpoint. Reason: %s.", reason)
    return {
        "agents_enabled": False,
        "disabled_at": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
    }


@router.post("/admin/enable")
def enable_agents() -> dict:
    """Re-enable agents after manual disable or auto-rollback investigation.

    Returns:
        Confirmation dictionary.
    """
    AutoRollbackMonitor.enable_agents()
    logger.info("Agent service RE-ENABLED via admin endpoint.")
    return {
        "agents_enabled": True,
        "enabled_at": datetime.now(timezone.utc).isoformat(),
    }


# ── Background helpers ─────────────────────────────────────────────────────────


def _check_accuracy_in_background() -> None:
    """Run accuracy check and trigger auto-rollback if needed.

    This runs as a FastAPI BackgroundTask so it doesn't block the response.
    The AutoRollbackMonitor updates the runtime state in-place.
    """
    try:
        status = _rollback_monitor.check()
        if status["rolled_back"]:
            logger.critical(
                "AUTO-ROLLBACK executed in background task. Reason: %s.",
                status.get("reason"),
            )
    except Exception as exc:
        logger.error("Background accuracy check failed: %s.", exc)


def _auto_document_id(text: str) -> str:
    """Generate a deterministic document ID from the text prefix.

    Args:
        text: Raw document text.

    Returns:
        A short slug usable as a document ID.
    """
    slug = text[:40].replace(" ", "_").replace("\n", "").replace("/", "-")
    ts = str(int(time.time()))[-6:]
    return f"doc_{slug}_{ts}"
