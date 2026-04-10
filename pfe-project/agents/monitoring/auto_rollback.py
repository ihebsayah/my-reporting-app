"""Auto-Rollback Monitor — disables agents automatically when quality drops.

Implements the plan's auto-rollback trigger:

    Condition 1: Agent accuracy < 85% (last 100 docs)
    AND
    Condition 2: Human override rate > 30%

    THEN:
      - Disable all agents (set AGENTS_ENABLED=False in runtime state)
      - Alert ops team (log CRITICAL)
      - System falls back to existing ML pipeline
      - Human investigates and re-enables

The rollback is runtime-only — it does not rewrite .env or restart the service.
Re-enable by calling enable_agents() after the issue is resolved.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from agents.config import get_agent_settings
from agents.monitoring.accuracy_tracker import AccuracyReport, AccuracyTracker

logger = logging.getLogger(__name__)
_settings = get_agent_settings()

# Runtime state — mutable, not persisted (service restart re-reads .env).
_agents_enabled_override: Optional[bool] = None
_last_rollback_reason: Optional[str] = None
_last_rollback_at: Optional[str] = None


class AutoRollbackMonitor:
    """Monitor that checks accuracy on a schedule and disables agents if needed.

    Usage::

        monitor = AutoRollbackMonitor()
        status = monitor.check()
        if status["rolled_back"]:
            # Notify ops team
    """

    def __init__(self) -> None:
        self.tracker = AccuracyTracker()
        logger.info("AutoRollbackMonitor initialised.")

    def check(self) -> dict:
        """Evaluate current agent accuracy and trigger rollback if needed.

        Returns:
            Status dictionary with ``rolled_back``, ``accuracy_report``, and reason.
        """
        global _agents_enabled_override, _last_rollback_reason, _last_rollback_at

        report: AccuracyReport = self.tracker.evaluate()

        if report.rollback_needed:
            reason = "; ".join(report.rollback_reasons)
            _agents_enabled_override = False
            _last_rollback_reason = reason
            _last_rollback_at = datetime.now(timezone.utc).isoformat()

            logger.critical(
                "AUTO-ROLLBACK TRIGGERED: Agents DISABLED. Reasons: %s. "
                "Reverting to existing ML pipeline. Investigate and call enable_agents().",
                reason,
            )

            return {
                "rolled_back": True,
                "reason": reason,
                "rolled_back_at": _last_rollback_at,
                "accuracy_report": report.__dict__,
                "agents_enabled": False,
            }

        return {
            "rolled_back": False,
            "reason": None,
            "accuracy_report": report.__dict__,
            "agents_enabled": self.agents_enabled(),
        }

    @staticmethod
    def agents_enabled() -> bool:
        """Return the effective agents-enabled state (config OR runtime override).

        Returns:
            True if agents are enabled, False if globally disabled or rolled back.
        """
        if _agents_enabled_override is not None:
            return _agents_enabled_override
        return _settings.agents_enabled

    @staticmethod
    def disable_agents(reason: str = "manual") -> None:
        """Manually disable agents at runtime (without restart).

        Args:
            reason: Human-readable reason for disabling.
        """
        global _agents_enabled_override, _last_rollback_reason, _last_rollback_at
        _agents_enabled_override = False
        _last_rollback_reason = reason
        _last_rollback_at = datetime.now(timezone.utc).isoformat()
        logger.warning("Agents MANUALLY DISABLED. Reason: %s.", reason)

    @staticmethod
    def enable_agents() -> None:
        """Re-enable agents after an investigation (clears runtime override).

        The config value (AGENTS_ENABLED env var) then takes effect again.
        """
        global _agents_enabled_override, _last_rollback_reason
        _agents_enabled_override = None
        _last_rollback_reason = None
        logger.info("Agents re-enabled (runtime override cleared).")

    @staticmethod
    def status() -> dict:
        """Return the current rollback status for the monitoring dashboard."""
        return {
            "agents_enabled": AutoRollbackMonitor.agents_enabled(),
            "rolled_back": _agents_enabled_override is False,
            "last_rollback_reason": _last_rollback_reason,
            "last_rollback_at": _last_rollback_at,
        }
