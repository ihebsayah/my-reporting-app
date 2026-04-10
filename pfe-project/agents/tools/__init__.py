"""Agent tools package — wraps existing ML models and DB for LangChain tool use."""

from agents.tools.db_tools import (
    lookup_vendor_history,
    get_document_history,
    save_agent_decision,
)
from agents.tools.memory_tools import (
    get_redis_pattern,
    set_redis_pattern,
    increment_redis_counter,
)
from agents.tools.ml_tools import (
    run_ner_extraction,
    run_confidence_scoring,
    run_bart_classification,
)

__all__ = [
    "get_document_history",
    "get_redis_pattern",
    "increment_redis_counter",
    "lookup_vendor_history",
    "run_bart_classification",
    "run_confidence_scoring",
    "run_ner_extraction",
    "save_agent_decision",
    "set_redis_pattern",
]
