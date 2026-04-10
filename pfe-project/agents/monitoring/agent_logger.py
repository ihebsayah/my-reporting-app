"""Structured agent logger — writes every agent interaction to a JSONL audit log.

Every document processing run produces one log entry with:
- Timestamp, document_id, session_id
- All four sub-agent results
- Final decision + confidence
- Human override (if any)
- Duration

Log file: artifacts/agent_logs/agent_decisions.jsonl
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_LOG_DIR = Path(os.environ.get("AGENT_LOG_DIR", "artifacts/agent_logs"))
_LOG_FILE = _LOG_DIR / "agent_decisions.jsonl"


class AgentLogger:
    """Writes structured JSONL audit logs for every agent interaction.

    Usage::

        agent_logger = AgentLogger()
        agent_logger.log_decision(
            document_id="doc_001",
            session_id="abc-123",
            final_decision="auto_approve",
            confidence=0.91,
            doc_type="invoice",
            agents_used=["classifier", "extractor", "validator", "router"],
            duration_ms=312,
        )
    """

    def __init__(self, log_dir: Optional[Path] = None) -> None:
        """Initialise the logger with an optional custom log directory.

        Args:
            log_dir: Override for the default log directory.
        """
        self.log_dir = log_dir or _LOG_DIR
        self.log_file = self.log_dir / "agent_decisions.jsonl"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        logger.info("AgentLogger writing to %s.", self.log_file)

    def log_decision(
        self,
        document_id: str,
        session_id: str,
        final_decision: str,
        confidence: float,
        doc_type: str,
        agents_used: List[str],
        duration_ms: int,
        extracted_fields: Optional[List[Dict[str, Any]]] = None,
        validation_issues: Optional[List[Dict[str, Any]]] = None,
        safety_rails_triggered: Optional[List[str]] = None,
        human_override: Optional[str] = None,
        fallback_used: bool = False,
        error: Optional[str] = None,
    ) -> None:
        """Append one structured decision record to the JSONL log.

        Args:
            document_id: Document being processed.
            session_id: Short-term memory session ID.
            final_decision: Final routing action.
            confidence: Final confidence score.
            doc_type: Classified document type.
            agents_used: List of agent names called.
            duration_ms: Total processing time in milliseconds.
            extracted_fields: List of extracted field dicts.
            validation_issues: List of validation issue dicts.
            safety_rails_triggered: List of triggered safety rail labels.
            human_override: Human's final decision if it differed.
            fallback_used: Whether the fallback path was used.
            error: Error message if fallback was triggered.
        """
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "document_id": document_id,
            "session_id": session_id,
            "agent": "master",
            "action": "process_document",
            "final_decision": final_decision,
            "confidence": confidence,
            "doc_type": doc_type,
            "sub_agents_called": agents_used,
            "duration_ms": duration_ms,
            "field_count": len(extracted_fields) if extracted_fields else 0,
            "validation_issue_count": len(validation_issues) if validation_issues else 0,
            "safety_rails_triggered": safety_rails_triggered or [],
            "human_override": human_override,
            "human_override_occurred": human_override is not None,
            "fallback_used": fallback_used,
            "error": error,
        }

        try:
            with self.log_file.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=True) + "\n")
        except Exception as exc:
            logger.error("AgentLogger failed to write record: %s.", exc)

    def log_feedback(
        self,
        document_id: str,
        agent_decision: str,
        human_outcome: str,
        vendor: str,
        amount: float,
        notes: Optional[str] = None,
    ) -> None:
        """Log a human feedback event.

        Args:
            document_id: The document that was reviewed.
            agent_decision: What the agent decided.
            human_outcome: What the human decided.
            vendor: Vendor name.
            amount: Invoice amount.
            notes: Optional human notes.
        """
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "human_feedback",
            "document_id": document_id,
            "agent_decision": agent_decision,
            "human_outcome": human_outcome,
            "vendor": vendor,
            "amount": amount,
            "notes": notes,
        }
        try:
            with self.log_file.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=True) + "\n")
        except Exception as exc:
            logger.error("AgentLogger.log_feedback failed: %s.", exc)
