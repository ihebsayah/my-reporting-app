"""Short-term agent memory — per-document conversation state.

Each document being processed gets its own ``ShortTermMemory`` instance
that tracks the agent conversation chain and intermediate results for
the lifetime of that single extraction run.  After the run completes
the memory can be serialised to Redis for auditing, then discarded.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from agents.config import get_agent_settings
from agents.memory.redis_client import get_redis_client

logger = logging.getLogger(__name__)
_settings = get_agent_settings()


class ShortTermMemory:
    """Holds the in-flight conversation state for one document extraction.

    Attributes:
        session_id: Unique ID for this extraction session.
        document_id: The document being processed.
        messages: Ordered list of agent messages/observations.
        context: Arbitrary key-value store for intermediate agent outputs.
    """

    def __init__(self, document_id: str) -> None:
        """Initialise short-term memory for a document.

        Args:
            document_id: Identifier for the document being extracted.
        """
        self.session_id: str = str(uuid.uuid4())
        self.document_id: str = document_id
        self.messages: List[Dict[str, Any]] = []
        self.context: Dict[str, Any] = {}
        self.started_at: str = datetime.now(timezone.utc).isoformat()
        logger.debug(
            "ShortTermMemory created: session_id=%s document_id=%s.",
            self.session_id, self.document_id,
        )

    def add_message(self, role: str, content: str, agent: Optional[str] = None) -> None:
        """Append a message to the conversation history.

        Args:
            role: ``"human"``, ``"ai"``, or ``"tool"``.
            content: Message text / tool output.
            agent: Optional agent name that produced this message.
        """
        msg = {
            "role": role,
            "content": content,
            "agent": agent,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.messages.append(msg)

    def set_context(self, key: str, value: Any) -> None:
        """Store an intermediate result in the shared context dict.

        Args:
            key: Context key (e.g. ``"classifier_result"``).
            value: Any JSON-serialisable value.
        """
        self.context[key] = value
        logger.debug("ShortTermMemory set_context key='%s'.", key)

    def get_context(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from the shared context dict.

        Args:
            key: Context key.
            default: Value to return if key is missing.

        Returns:
            Stored value or default.
        """
        return self.context.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the memory state to a plain dictionary.

        Returns:
            Dictionary representation of this memory instance.
        """
        return {
            "session_id": self.session_id,
            "document_id": self.document_id,
            "started_at": self.started_at,
            "message_count": len(self.messages),
            "messages": self.messages,
            "context": self.context,
        }

    def persist_to_redis(self) -> bool:
        """Write the full session to Redis for auditing/debugging.

        The key is ``session:{session_id}`` with the configured short-term TTL.

        Returns:
            True on success, False if Redis is unavailable or write fails.
        """
        client = get_redis_client()
        if client is None:
            return False
        try:
            key = f"session:{self.session_id}"
            client.setex(key, _settings.redis_ttl_seconds, json.dumps(self.to_dict()))
            logger.debug("ShortTermMemory persisted to Redis key='%s'.", key)
            return True
        except Exception as exc:
            logger.warning("ShortTermMemory.persist_to_redis failed: %s", exc)
            return False
