"""Redis memory tools — LangChain-compatible helpers for real-time pattern storage.

Redis is used for:
- Short-term conversation state (current document context)
- Learned patterns (vendor → outcome, amount_band → risk level)
- Counters / accuracy windows for monitoring
"""

import json
import logging
from typing import Any, Dict, Optional

from langchain_core.tools import tool

from agents.config import get_agent_settings

logger = logging.getLogger(__name__)
_settings = get_agent_settings()

# Module-level Redis client — created lazily.
_redis_client = None


def _get_redis():
    """Return a Redis client, creating it on first call.

    Falls back gracefully if redis is not available so the agent service
    can still start in environments without Redis.
    """
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    try:
        import redis  # type: ignore

        _redis_client = redis.from_url(
            _settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=3,
            socket_timeout=3,
        )
        _redis_client.ping()
        logger.info("Redis connected at %s.", _settings.redis_url)
    except Exception as exc:
        logger.warning("Redis unavailable (%s) — memory tools will no-op.", exc)
        _redis_client = None
    return _redis_client


@tool
def get_redis_pattern(pattern_key: str) -> str:
    """Retrieve a learned pattern from Redis by key.

    Args:
        pattern_key: Redis key (e.g. ``vendor:acme:outcome``).

    Returns:
        JSON string with the pattern value, or ``{"found": false}`` if absent.
    """
    logger.debug("get_redis_pattern key='%s'.", pattern_key)
    client = _get_redis()
    if client is None:
        return json.dumps({"found": False, "reason": "redis_unavailable"})
    try:
        value = client.get(pattern_key)
        if value is None:
            return json.dumps({"found": False, "key": pattern_key})
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = value
        return json.dumps({"found": True, "key": pattern_key, "value": parsed})
    except Exception as exc:
        logger.error("get_redis_pattern failed: %s", exc)
        return json.dumps({"found": False, "error": str(exc)})


@tool
def set_redis_pattern(pattern_key: str, pattern_value: str, ttl_seconds: int = 0) -> str:
    """Store a learned pattern in Redis.

    Args:
        pattern_key: Redis key.
        pattern_value: Value to store (will be stored as a JSON string).
        ttl_seconds: Optional TTL in seconds; 0 uses the default from config.

    Returns:
        JSON string confirming success or failure.
    """
    logger.info("set_redis_pattern key='%s'.", pattern_key)
    client = _get_redis()
    if client is None:
        return json.dumps({"success": False, "reason": "redis_unavailable"})
    ttl = ttl_seconds if ttl_seconds > 0 else _settings.redis_pattern_ttl_seconds
    try:
        # Ensure the value is a JSON string.
        if not isinstance(pattern_value, str):
            pattern_value = json.dumps(pattern_value)
        client.setex(pattern_key, ttl, pattern_value)
        return json.dumps({"success": True, "key": pattern_key, "ttl": ttl})
    except Exception as exc:
        logger.error("set_redis_pattern failed: %s", exc)
        return json.dumps({"success": False, "error": str(exc)})


@tool
def increment_redis_counter(counter_key: str, amount: int = 1, ttl_seconds: int = 0) -> str:
    """Atomically increment a Redis counter.

    Used by the monitoring module to track accuracy windows
    without race conditions.

    Args:
        counter_key: Redis key for the counter.
        amount: Value to add (default 1).
        ttl_seconds: Optional TTL; 0 uses the default config TTL.

    Returns:
        JSON string with the new counter value.
    """
    client = _get_redis()
    if client is None:
        return json.dumps({"success": False, "reason": "redis_unavailable"})
    ttl = ttl_seconds if ttl_seconds > 0 else _settings.redis_ttl_seconds
    try:
        new_value = client.incrby(counter_key, amount)
        client.expire(counter_key, ttl)
        return json.dumps({"success": True, "key": counter_key, "new_value": new_value})
    except Exception as exc:
        logger.error("increment_redis_counter failed: %s", exc)
        return json.dumps({"success": False, "error": str(exc)})


# ── Low-level helpers (not exposed as LangChain tools) ────────────────────────


def store_feedback_pattern(
    vendor: str,
    amount: float,
    outcome: str,
    agent_decision: str,
) -> None:
    """Record a human-feedback learning signal in Redis.

    Called by the master agent after human feedback is received.
    Stores a rich pattern object and increments per-outcome vendor counters.

    Args:
        vendor: Vendor name (lowercased as key part).
        amount: Invoice amount.
        outcome: Human's final outcome (approve / reject / review).
        agent_decision: What the agent had originally decided.
    """
    client = _get_redis()
    if client is None:
        return
    vendor_key = vendor.lower().replace(" ", "_")
    amount_band = _amount_band(amount)

    pattern_key = f"pattern:vendor:{vendor_key}:outcome"
    existing_raw = client.get(pattern_key)
    if existing_raw:
        try:
            existing = json.loads(existing_raw)
        except json.JSONDecodeError:
            existing = {}
    else:
        existing = {}

    existing[outcome] = existing.get(outcome, 0) + 1
    existing["last_amount"] = amount
    existing["last_agent_decision"] = agent_decision
    existing["total"] = sum(v for k, v in existing.items() if k in ("approve", "reject", "review"))

    ttl = _settings.redis_pattern_ttl_seconds
    client.setex(pattern_key, ttl, json.dumps(existing))

    # Per amount-band counter.
    band_key = f"pattern:amount_band:{amount_band}:{outcome}"
    client.incrby(band_key, 1)
    client.expire(band_key, ttl)

    logger.info(
        "Stored feedback pattern: vendor=%s, outcome=%s, amount_band=%s.",
        vendor_key, outcome, amount_band,
    )


def get_vendor_pattern(vendor: str) -> Optional[Dict[str, Any]]:
    """Retrieve the learned outcome pattern for a vendor.

    Args:
        vendor: Vendor name to look up.

    Returns:
        Dictionary of outcome counts, or None if not found.
    """
    client = _get_redis()
    if client is None:
        return None
    vendor_key = vendor.lower().replace(" ", "_")
    raw = client.get(f"pattern:vendor:{vendor_key}:outcome")
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _amount_band(amount: float) -> str:
    """Map a numeric amount to a discrete band label.

    Args:
        amount: Invoice total amount.

    Returns:
        String band label (e.g. ``"1k-5k"``).
    """
    if amount < 1_000:
        return "lt1k"
    if amount < 5_000:
        return "1k-5k"
    if amount < 20_000:
        return "5k-20k"
    if amount < 100_000:
        return "20k-100k"
    return "gt100k"
