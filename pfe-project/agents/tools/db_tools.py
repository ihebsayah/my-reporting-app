"""Database tools — agent-side PostgreSQL/SQLite queries for vendor history and pattern storage.

These functions are intentionally read-only for vendor lookups.  Writes
(saving agent decisions) go through a dedicated agent_decisions table so
the existing extraction result schema stays untouched.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from agents.config import get_agent_settings

logger = logging.getLogger(__name__)
_settings = get_agent_settings()

# Module-level engine — created lazily on first use.
_engine = None
_SessionLocal = None


def _get_session():
    """Return a SQLAlchemy session, creating the engine on first call."""
    global _engine, _SessionLocal
    if _engine is None:
        _engine = create_engine(
            _settings.database_url,
            connect_args={"check_same_thread": False}
            if "sqlite" in _settings.database_url
            else {},
        )
        _SessionLocal = sessionmaker(bind=_engine)
        _ensure_agent_tables(_engine)
    return _SessionLocal()


def _ensure_agent_tables(engine) -> None:
    """Create agent-specific tables if they do not exist yet."""
    ddl = """
    CREATE TABLE IF NOT EXISTS agent_decisions (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id TEXT        NOT NULL,
        action      TEXT        NOT NULL,
        confidence  REAL        NOT NULL,
        reasoning   TEXT,
        human_override TEXT,
        human_feedback TEXT,
        created_at  TEXT        NOT NULL
    );

    CREATE TABLE IF NOT EXISTS agent_patterns (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        pattern_key TEXT        NOT NULL UNIQUE,
        pattern_value TEXT      NOT NULL,
        occurrences INTEGER     DEFAULT 1,
        last_seen   TEXT        NOT NULL,
        created_at  TEXT        NOT NULL
    );
    """
    with engine.connect() as conn:
        for statement in ddl.strip().split(";"):
            stmt = statement.strip()
            if stmt:
                conn.execute(text(stmt))
        conn.commit()
    logger.info("Agent DB tables ensured.")


@tool
def lookup_vendor_history(vendor_name: str) -> str:
    """Look up historical invoice data for a vendor from the extraction results DB.

    Queries the existing ``extraction_result_fields`` table (written by the
    FastAPI pipeline) to understand average amounts, decision distribution,
    and whether the vendor has been seen before.

    Args:
        vendor_name: Vendor name string to search for.

    Returns:
        JSON string with vendor history summary or not-found indicator.
    """
    logger.info("lookup_vendor_history called for vendor='%s'.", vendor_name)
    try:
        session = _get_session()
        query = text(
            """
            SELECT
                COUNT(*)                as total_invoices,
                AVG(CAST(erf.value AS REAL)) as avg_amount,
                MIN(CAST(erf.value AS REAL)) as min_amount,
                MAX(CAST(erf.value AS REAL)) as max_amount,
                SUM(CASE WHEN er.overall_decision = 'auto' THEN 1 ELSE 0 END)   as auto_count,
                SUM(CASE WHEN er.overall_decision = 'review' THEN 1 ELSE 0 END) as review_count,
                SUM(CASE WHEN er.overall_decision = 'reject' THEN 1 ELSE 0 END) as reject_count
            FROM extraction_results er
            JOIN extraction_result_fields erf
                ON erf.extraction_result_id = er.id
                AND erf.field_name = 'VENDOR_NAME'
            WHERE LOWER(erf.value) LIKE LOWER(:vendor_pattern)
            """
        )
        result = session.execute(query, {"vendor_pattern": f"%{vendor_name}%"}).fetchone()

        if result is None or result[0] == 0:
            logger.info("No history found for vendor='%s'.", vendor_name)
            return json.dumps({
                "vendor": vendor_name,
                "found": False,
                "total_invoices": 0,
                "is_new_vendor": True,
            })

        total = result[0] or 0
        output = {
            "vendor": vendor_name,
            "found": total > 0,
            "total_invoices": total,
            "avg_amount": round(result[1] or 0, 2),
            "min_amount": round(result[2] or 0, 2),
            "max_amount": round(result[3] or 0, 2),
            "auto_count": result[4] or 0,
            "review_count": result[5] or 0,
            "reject_count": result[6] or 0,
            "is_new_vendor": total == 0,
            "auto_rate": round((result[4] or 0) / total, 3) if total > 0 else 0,
        }
        logger.info("Vendor history for '%s': %d invoices found.", vendor_name, total)
        return json.dumps(output)
    except Exception as exc:
        logger.error("lookup_vendor_history failed: %s", exc)
        return json.dumps({"vendor": vendor_name, "found": False, "error": str(exc), "is_new_vendor": True})
    finally:
        session.close()


@tool
def get_document_history(document_id: str) -> str:
    """Retrieve past extraction results for a specific document ID.

    Args:
        document_id: The document identifier to look up.

    Returns:
        JSON string with a list of past decisions for the document.
    """
    logger.info("get_document_history called for document_id='%s'.", document_id)
    try:
        session = _get_session()
        query = text(
            """
            SELECT overall_decision, scorer, model_version, processed_at
            FROM extraction_results
            WHERE document_id = :doc_id
            ORDER BY processed_at DESC
            LIMIT 10
            """
        )
        rows = session.execute(query, {"doc_id": document_id}).fetchall()
        history = [
            {
                "decision": row[0],
                "scorer": row[1],
                "model_version": row[2],
                "processed_at": row[3],
            }
            for row in rows
        ]
        return json.dumps({"document_id": document_id, "history": history, "count": len(history)})
    except Exception as exc:
        logger.error("get_document_history failed: %s", exc)
        return json.dumps({"document_id": document_id, "history": [], "error": str(exc)})
    finally:
        session.close()


def save_agent_decision(
    document_id: str,
    action: str,
    confidence: float,
    reasoning: str,
    human_override: Optional[str] = None,
    human_feedback: Optional[str] = None,
) -> bool:
    """Persist an agent decision record to the agent_decisions table.

    This is called programmatically (not as a LangChain tool) by the
    master agent after completing a document workflow.

    Args:
        document_id: Document being processed.
        action: Agent routing decision (auto_approve / human_review / reject).
        confidence: Overall confidence score (0.0 – 1.0).
        reasoning: Agent's explainability text.
        human_override: Human's final decision if it differed from agent.
        human_feedback: Free-text feedback from the human reviewer.

    Returns:
        True on success, False on failure.
    """
    logger.info(
        "save_agent_decision: document_id=%s action=%s confidence=%.3f",
        document_id, action, confidence,
    )
    try:
        session = _get_session()
        now = datetime.now(timezone.utc).isoformat()
        session.execute(
            text(
                """
                INSERT INTO agent_decisions
                    (document_id, action, confidence, reasoning, human_override, human_feedback, created_at)
                VALUES
                    (:document_id, :action, :confidence, :reasoning, :human_override, :human_feedback, :created_at)
                """
            ),
            {
                "document_id": document_id,
                "action": action,
                "confidence": confidence,
                "reasoning": reasoning,
                "human_override": human_override,
                "human_feedback": human_feedback,
                "created_at": now,
            },
        )
        session.commit()
        logger.info("Agent decision saved for document_id=%s.", document_id)
        return True
    except Exception as exc:
        logger.error("save_agent_decision failed: %s", exc)
        return False
    finally:
        session.close()


def get_agent_accuracy_stats(window: int = 100) -> Dict[str, Any]:
    """Return accuracy statistics for the last N agent decisions.

    Used by the auto-rollback monitor.

    Args:
        window: Number of most recent decisions to analyse.

    Returns:
        Dictionary with accuracy, override_rate, and decision counts.
    """
    try:
        session = _get_session()
        query = text(
            """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN human_override IS NULL THEN 1 ELSE 0 END) as agreed,
                SUM(CASE WHEN human_override IS NOT NULL THEN 1 ELSE 0 END) as overridden
            FROM (
                SELECT human_override
                FROM agent_decisions
                ORDER BY created_at DESC
                LIMIT :window
            )
            """
        )
        row = session.execute(query, {"window": window}).fetchone()
        if row is None or row[0] == 0:
            return {"total": 0, "accuracy": 1.0, "override_rate": 0.0}
        total = row[0]
        agreed = row[1] or 0
        overridden = row[2] or 0
        return {
            "total": total,
            "agreed": agreed,
            "overridden": overridden,
            "accuracy": round(agreed / total, 4),
            "override_rate": round(overridden / total, 4),
        }
    except Exception as exc:
        logger.error("get_agent_accuracy_stats failed: %s", exc)
        return {"total": 0, "accuracy": 1.0, "override_rate": 0.0, "error": str(exc)}
    finally:
        session.close()
