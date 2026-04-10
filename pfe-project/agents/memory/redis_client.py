"""Redis client singleton for the agent memory layer."""

import logging
from typing import Optional

from agents.config import get_agent_settings

logger = logging.getLogger(__name__)
_settings = get_agent_settings()

_client = None


def get_redis_client():
    """Return a shared Redis client instance, or None if Redis is unavailable.

    Returns:
        A ``redis.Redis`` instance or ``None``.
    """
    global _client
    if _client is not None:
        return _client
    try:
        import redis  # type: ignore

        _client = redis.from_url(
            _settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=3,
            socket_timeout=3,
        )
        _client.ping()
        logger.info("Redis client initialized at %s.", _settings.redis_url)
    except Exception as exc:
        logger.warning("Redis unavailable — short-term memory will be in-process only. (%s)", exc)
        _client = None
    return _client
