"""Long-term agent memory — cross-document learning via SQLite + Redis.

This module handles:
1. Auto-creating the ``agent_patterns`` table on first use (no migration needed).
2. Accumulating vendor approve/reject/review statistics per pattern key.
3. Querying historical patterns so agents can flag risky vendors in real-time.
4. Writing to Redis for sub-millisecond reads; falling back to SQLite on miss.

Design principles
-----------------
- **Self-healing**: the table is created if it doesn't exist, so no separate
  migration is required — the memory layer just works.
- **Accumulated stats**: each ``record_feedback`` call increments counters
  rather than overwriting, so ``should_flag_vendor`` gets real rates.
- **Cached engine**: a single SQLAlchemy engine is created per database URL
  and reused across calls to avoid connection leaks.
"""

import json
import logging
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Dict, Optional

from agents.config import get_agent_settings
from agents.memory.redis_client import get_redis_client
from agents.tools.memory_tools import store_feedback_pattern, get_vendor_pattern

logger = logging.getLogger(__name__)
_settings = get_agent_settings()

# ── Engine singleton (one per DB URL) ─────────────────────────────────────────

@lru_cache(maxsize=4)
def _get_engine(database_url: str):
    """Return a cached SQLAlchemy engine for the given URL."""
    from sqlalchemy import create_engine
    connect_args = {"check_same_thread": False} if "sqlite" in database_url else {}
    return create_engine(database_url, connect_args=connect_args)


def _ensure_table(engine) -> None:
    """Verify agent_patterns exists; if not, create it.

    The canonical schema is owned by ``agents/tools/db_tools.py``.
    This function is a safety net in case LongTermMemory is used outside
    the agent service (e.g. in standalone scripts).
    """
    from sqlalchemy import text
    ddl = """
    CREATE TABLE IF NOT EXISTS agent_patterns (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        pattern_key         TEXT    NOT NULL UNIQUE,
        approve_count       INTEGER NOT NULL DEFAULT 0,
        reject_count        INTEGER NOT NULL DEFAULT 0,
        review_count        INTEGER NOT NULL DEFAULT 0,
        total_count         INTEGER NOT NULL DEFAULT 0,
        last_amount         REAL,
        last_outcome        TEXT,
        last_agent_decision TEXT,
        last_document_id    TEXT,
        notes               TEXT,
        pattern_value       TEXT,
        occurrences         INTEGER NOT NULL DEFAULT 0,
        last_seen           TEXT,
        created_at          TEXT
    )
    """
    migrate_columns = [
        "ALTER TABLE agent_patterns ADD COLUMN approve_count       INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE agent_patterns ADD COLUMN reject_count        INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE agent_patterns ADD COLUMN review_count        INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE agent_patterns ADD COLUMN total_count         INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE agent_patterns ADD COLUMN last_amount         REAL",
        "ALTER TABLE agent_patterns ADD COLUMN last_outcome        TEXT",
        "ALTER TABLE agent_patterns ADD COLUMN last_agent_decision TEXT",
        "ALTER TABLE agent_patterns ADD COLUMN last_document_id    TEXT",
        "ALTER TABLE agent_patterns ADD COLUMN notes               TEXT",
    ]
    with engine.connect() as conn:
        conn.execute(text(ddl))
        for col_sql in migrate_columns:
            try:
                conn.execute(text(col_sql))
            except Exception:
                pass  # Column already exists.
        conn.commit()


# ── LongTermMemory ─────────────────────────────────────────────────────────────

class LongTermMemory:
    """Cross-document learning memory backed by SQLite (durable) and Redis (fast).

    Usage::

        ltm = LongTermMemory()
        ltm.record_feedback(
            document_id="doc_123",
            vendor="Acme Corp",
            amount=5000.0,
            agent_decision="auto_approve",
            human_outcome="approve",
        )
        pattern = ltm.lookup_vendor_pattern("Acme Corp")
        flagged = ltm.should_flag_vendor("New Vendor SARL")  # True — unknown
    """

    def __init__(self) -> None:
        self._engine = _get_engine(_settings.database_url)
        try:
            _ensure_table(self._engine)
        except Exception as exc:
            logger.warning("LongTermMemory: could not ensure agent_patterns table: %s", exc)

    # ── Public API ─────────────────────────────────────────────────────────────

    def record_feedback(
        self,
        document_id: str,
        vendor: str,
        amount: float,
        agent_decision: str,
        human_outcome: str,
        notes: Optional[str] = None,
    ) -> None:
        """Record a human feedback signal and accumulate learned statistics.

        Args:
            document_id: Document that was processed.
            vendor: Extracted vendor name.
            amount: Extracted invoice amount.
            agent_decision: What the agent originally decided.
            human_outcome: What the human decided (approve / reject / review).
            notes: Optional reviewer notes.
        """
        logger.info(
            "LongTermMemory.record_feedback: doc=%s vendor=%s agent=%s human=%s",
            document_id, vendor, agent_decision, human_outcome,
        )

        # 1. Write to Redis for immediate in-session availability.
        try:
            store_feedback_pattern(
                vendor=vendor,
                amount=amount,
                outcome=human_outcome,
                agent_decision=agent_decision,
            )
        except Exception as exc:
            logger.debug("LongTermMemory: Redis write skipped (%s).", exc)

        # 2. Accumulate stats in SQLite (persistent across restarts).
        self._accumulate_pattern(
            pattern_key=f"vendor:{vendor.lower()}",
            vendor=vendor,
            amount=amount,
            agent_decision=agent_decision,
            human_outcome=human_outcome,
            document_id=document_id,
            notes=notes,
        )

    def lookup_vendor_pattern(self, vendor: str) -> Optional[Dict[str, Any]]:
        """Look up the accumulated outcome statistics for a vendor.

        Checks Redis first (fast); falls back to SQLite if not cached.

        Args:
            vendor: Vendor name.

        Returns:
            Stats dict with approve_count, reject_count, total_count, etc.
            or None if no history exists.
        """
        # Try Redis first (sub-ms).
        try:
            pattern = get_vendor_pattern(vendor)
            if pattern:
                logger.debug("LongTermMemory: vendor pattern in Redis for '%s'.", vendor)
                return pattern
        except Exception:
            pass

        # Fall back to SQLite.
        return self._get_db_pattern(f"vendor:{vendor.lower()}")

    def should_flag_vendor(self, vendor: str) -> bool:
        """Return True if this vendor should be treated with extra scrutiny.

        Logic:
        - Unknown vendor (no history) → True (RAIL_2 already handles this).
        - Reject rate > 30% → True.
        - Total < 3 observations → True (insufficient data).

        Args:
            vendor: Vendor name.

        Returns:
            True if the vendor should be flagged for human review.
        """
        pattern = self.lookup_vendor_pattern(vendor)
        if not pattern:
            return True  # No history — flag as unknown.

        total = pattern.get("total_count", pattern.get("total", 0))
        if total < 3:
            return True  # Insufficient data.

        reject = pattern.get("reject_count", pattern.get("reject", 0))
        reject_rate = reject / total
        logger.debug(
            "should_flag_vendor '%s': reject_rate=%.0f%% total=%d.",
            vendor, reject_rate * 100, total,
        )
        return reject_rate > 0.30  # Flag if > 30% rejection rate.

    def get_vendor_summary(self, vendor: str) -> str:
        """Return a human-readable summary string for the LLM validator prompt.

        Args:
            vendor: Vendor name.

        Returns:
            One-line summary describing vendor history, or 'no history found'.
        """
        pattern = self.lookup_vendor_pattern(vendor)
        if not pattern:
            return "no history found"
        total   = pattern.get("total_count", pattern.get("total", 0))
        approve = pattern.get("approve_count", pattern.get("approve", 0))
        reject  = pattern.get("reject_count", pattern.get("reject", 0))
        review  = pattern.get("review_count", pattern.get("review", 0))
        last    = pattern.get("last_outcome", "unknown")
        return (
            f"{total} past decision(s): {approve} approved, {reject} rejected, "
            f"{review} reviewed. Last outcome: {last}."
        )

    # ── SQLite helpers ─────────────────────────────────────────────────────────

    def _accumulate_pattern(
        self,
        pattern_key: str,
        vendor: str,
        amount: float,
        agent_decision: str,
        human_outcome: str,
        document_id: str,
        notes: Optional[str],
    ) -> None:
        """Insert or update the accumulated stats row for a pattern key."""
        from sqlalchemy import text

        # Map outcome to column name
        outcome_col_map = {
            "approve": "approve_count",
            "approved": "approve_count",
            "auto": "approve_count",
            "reject": "reject_count",
            "rejected": "reject_count",
            "review": "review_count",
            "human_review": "review_count",
        }
        col = outcome_col_map.get(human_outcome.lower(), "review_count")
        now = datetime.now(timezone.utc).isoformat()

        try:
            with self._engine.connect() as conn:
                existing = conn.execute(
                    text(
                        "SELECT id, approve_count, reject_count, review_count, total_count "
                        "FROM agent_patterns WHERE pattern_key = :k"
                    ),
                    {"k": pattern_key},
                ).fetchone()

                if existing:
                    row_id, ap, rej, rev, tot = existing
                    new_ap  = ap  + (1 if col == "approve_count" else 0)
                    new_rej = rej + (1 if col == "reject_count"  else 0)
                    new_rev = rev + (1 if col == "review_count"  else 0)
                    conn.execute(
                        text(
                            "UPDATE agent_patterns SET "
                            "approve_count=:ap, reject_count=:rej, review_count=:rev, "
                            "total_count=:tot, last_amount=:la, last_outcome=:lo, "
                            "last_agent_decision=:lad, last_document_id=:ldoc, "
                            "notes=:n, occurrences=occurrences+1, last_seen=:ls "
                            "WHERE pattern_key=:k"
                        ),
                        {
                            "ap": new_ap, "rej": new_rej, "rev": new_rev,
                            "tot": tot + 1, "la": amount, "lo": human_outcome,
                            "lad": agent_decision, "ldoc": document_id,
                            "n": notes, "ls": now, "k": pattern_key,
                        },
                    )
                else:
                    conn.execute(
                        text(
                            "INSERT INTO agent_patterns "
                            "(pattern_key, approve_count, reject_count, review_count, "
                            "total_count, last_amount, last_outcome, last_agent_decision, "
                            "last_document_id, notes, occurrences, last_seen, created_at) "
                            "VALUES (:k, :ap, :rej, :rev, 1, :la, :lo, :lad, :ldoc, :n, 1, :ls, :ca)"
                        ),
                        {
                            "k": pattern_key,
                            "ap": 1 if col == "approve_count" else 0,
                            "rej": 1 if col == "reject_count" else 0,
                            "rev": 1 if col == "review_count" else 0,
                            "la": amount, "lo": human_outcome,
                            "lad": agent_decision, "ldoc": document_id,
                            "n": notes, "ls": now, "ca": now,
                        },
                    )
                conn.commit()
                logger.debug("LongTermMemory: pattern upserted for key='%s'.", pattern_key)
        except Exception as exc:
            logger.error("LongTermMemory._accumulate_pattern failed for '%s': %s", pattern_key, exc)

    def _get_db_pattern(self, pattern_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve accumulated stats from SQLite for a pattern key."""
        from sqlalchemy import text

        try:
            with self._engine.connect() as conn:
                row = conn.execute(
                    text(
                        "SELECT approve_count, reject_count, review_count, total_count, "
                        "last_amount, last_outcome, last_agent_decision, last_document_id "
                        "FROM agent_patterns WHERE pattern_key = :k"
                    ),
                    {"k": pattern_key},
                ).fetchone()
            if row:
                return {
                    "approve_count": row[0],
                    "reject_count":  row[1],
                    "review_count":  row[2],
                    "total_count":   row[3],
                    "last_amount":   row[4],
                    "last_outcome":  row[5],
                    "last_agent_decision": row[6],
                    "last_document_id":    row[7],
                }
        except Exception as exc:
            logger.error("LongTermMemory._get_db_pattern failed for '%s': %s", pattern_key, exc)
        return None
