"""Long-term agent memory — cross-document learning via Redis + PostgreSQL.

This module handles:
1. Recording human feedback and storing it as learnable patterns in Redis.
2. Aggregating patterns into the PostgreSQL ``agent_patterns`` table for
   persistence across Redis restarts.
3. Querying historical patterns to inform future agent decisions.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from agents.config import get_agent_settings
from agents.memory.redis_client import get_redis_client
from agents.tools.memory_tools import store_feedback_pattern, get_vendor_pattern

logger = logging.getLogger(__name__)
_settings = get_agent_settings()


class LongTermMemory:
    """Cross-document learning memory backed by Redis (fast) and PostgreSQL (durable).

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
    """

    def record_feedback(
        self,
        document_id: str,
        vendor: str,
        amount: float,
        agent_decision: str,
        human_outcome: str,
        notes: Optional[str] = None,
    ) -> None:
        """Record a human feedback signal and update learned patterns.

        Called immediately after the human makes a decision so agents can
        learn in real-time.

        Args:
            document_id: Document that was processed.
            vendor: Extracted vendor name.
            amount: Extracted invoice amount.
            agent_decision: What the agent originally decided.
            human_outcome: What the human actually decided (approve/reject/review).
            notes: Optional human reviewer notes.
        """
        logger.info(
            "LongTermMemory.record_feedback: doc=%s vendor=%s agent=%s human=%s",
            document_id, vendor, agent_decision, human_outcome,
        )

        # 1. Write to Redis for immediate availability.
        store_feedback_pattern(
            vendor=vendor,
            amount=amount,
            outcome=human_outcome,
            agent_decision=agent_decision,
        )

        # 2. Persist to PostgreSQL agent_patterns table for durability.
        self._upsert_pg_pattern(
            pattern_key=f"vendor:{vendor.lower()}",
            pattern_value={
                "vendor": vendor,
                "last_outcome": human_outcome,
                "last_agent_decision": agent_decision,
                "last_amount": amount,
                "document_id": document_id,
                "notes": notes,
            },
        )

    def lookup_vendor_pattern(self, vendor: str) -> Optional[Dict[str, Any]]:
        """Look up the learned outcome history for a vendor.

        Checks Redis first (fast); falls back to PostgreSQL if not in Redis.

        Args:
            vendor: Vendor name.

        Returns:
            Pattern dictionary or None if no history exists.
        """
        # Try Redis first.
        pattern = get_vendor_pattern(vendor)
        if pattern:
            logger.debug("LongTermMemory: vendor pattern found in Redis for '%s'.", vendor)
            return pattern

        # Fall back to PostgreSQL.
        pattern = self._get_pg_pattern(f"vendor:{vendor.lower()}")
        if pattern:
            logger.debug("LongTermMemory: vendor pattern found in PostgreSQL for '%s'.", vendor)
        return pattern

    def should_flag_vendor(self, vendor: str) -> bool:
        """Determine if this vendor should be flagged for extra scrutiny.

        A vendor is flagged if it has been rejected more than it has been
        approved in the last N interactions stored in Redis.

        Args:
            vendor: Vendor name.

        Returns:
            True if the vendor should be treated with extra caution.
        """
        pattern = self.lookup_vendor_pattern(vendor)
        if not pattern:
            return True  # Unknown vendor → flag by default (safety rail 2).
        reject = pattern.get("reject", 0)
        approve = pattern.get("approve", 0)
        total = pattern.get("total", 0)
        if total == 0:
            return True
        reject_rate = reject / total
        logger.debug(
            "should_flag_vendor '%s': reject_rate=%.2f total=%d.", vendor, reject_rate, total
        )
        return reject_rate > 0.3  # Flag if rejection rate > 30%.

    # ── PostgreSQL helpers ─────────────────────────────────────────────────

    def _upsert_pg_pattern(self, pattern_key: str, pattern_value: Dict[str, Any]) -> None:
        """Upsert a pattern record into the agent_patterns table."""
        try:
            from sqlalchemy import create_engine, text

            engine = create_engine(
                _settings.database_url,
                connect_args={"check_same_thread": False}
                if "sqlite" in _settings.database_url
                else {},
            )
            now = datetime.now(timezone.utc).isoformat()
            value_json = json.dumps(pattern_value)
            with engine.connect() as conn:
                existing = conn.execute(
                    text("SELECT id, occurrences FROM agent_patterns WHERE pattern_key = :k"),
                    {"k": pattern_key},
                ).fetchone()
                if existing:
                    conn.execute(
                        text(
                            "UPDATE agent_patterns "
                            "SET pattern_value = :v, occurrences = :o, last_seen = :ls "
                            "WHERE pattern_key = :k"
                        ),
                        {"v": value_json, "o": (existing[1] or 0) + 1, "ls": now, "k": pattern_key},
                    )
                else:
                    conn.execute(
                        text(
                            "INSERT INTO agent_patterns (pattern_key, pattern_value, occurrences, last_seen, created_at) "
                            "VALUES (:k, :v, 1, :ls, :ca)"
                        ),
                        {"k": pattern_key, "v": value_json, "ls": now, "ca": now},
                    )
                conn.commit()
        except Exception as exc:
            logger.error("_upsert_pg_pattern failed for key='%s': %s", pattern_key, exc)

    def _get_pg_pattern(self, pattern_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve a pattern from the agent_patterns table."""
        try:
            from sqlalchemy import create_engine, text

            engine = create_engine(
                _settings.database_url,
                connect_args={"check_same_thread": False}
                if "sqlite" in _settings.database_url
                else {},
            )
            with engine.connect() as conn:
                row = conn.execute(
                    text("SELECT pattern_value FROM agent_patterns WHERE pattern_key = :k"),
                    {"k": pattern_key},
                ).fetchone()
            if row:
                return json.loads(row[0])
        except Exception as exc:
            logger.error("_get_pg_pattern failed for key='%s': %s", pattern_key, exc)
        return None
