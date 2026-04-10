"""Agent API package — FastAPI router for the agent service endpoints."""

from agents.api.extraction_agent import router as agent_router

__all__ = ["agent_router"]
