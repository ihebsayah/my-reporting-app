"""Accuracy Tracker — measures agent decision quality over a sliding window.

Tracks:
- Agreement rate (agent decision == human decision).
- Override rate (human overrode agent decision).
- Per-action breakdown (auto_approve, human_review, reject).
- Rolling window of the last N decisions for auto-rollback.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict

from agents.config import get_agent_settings
from agents.tools.db_tools import get_agent_accuracy_stats

logger = logging.getLogger(__name__)
_settings = get_agent_settings()


@dataclass
class AccuracyReport:
    """Snapshot of agent accuracy for a given window."""

    window: int
    total_decisions: int
    agreed: int
    overridden: int
    accuracy: float
    override_rate: float
    rollback_needed: bool
    rollback_reasons: list


class AccuracyTracker:
    """Tracks agent decision accuracy and recommends auto-rollback if needed.

    Reads directly from the ``agent_decisions`` table populated by the
    Master Agent after each document run.

    Usage::

        tracker = AccuracyTracker()
        report = tracker.evaluate()
        if report.rollback_needed:
            # Disable agents
    """

    def __init__(self, window: int = None) -> None:
        """Initialise with optional window override.

        Args:
            window: How many recent decisions to evaluate. Defaults to config value.
        """
        self.window = window or _settings.rollback_window_size
        logger.info("AccuracyTracker initialised with window=%d.", self.window)

    def evaluate(self) -> AccuracyReport:
        """Compute accuracy stats and determine if auto-rollback should trigger.

        Returns:
            AccuracyReport with accuracy, override_rate, and rollback recommendation.
        """
        stats = get_agent_accuracy_stats(window=self.window)
        total = stats.get("total", 0)
        accuracy = stats.get("accuracy", 1.0)
        override_rate = stats.get("override_rate", 0.0)

        rollback_reasons = []
        rollback_needed = False

        if total >= self.window:  # Only evaluate when we have enough data.
            if accuracy < _settings.rollback_accuracy_threshold:
                rollback_reasons.append(
                    f"Accuracy {accuracy:.1%} < threshold {_settings.rollback_accuracy_threshold:.1%}"
                )
            if override_rate > _settings.rollback_override_rate_threshold:
                rollback_reasons.append(
                    f"Override rate {override_rate:.1%} > threshold {_settings.rollback_override_rate_threshold:.1%}"
                )
            rollback_needed = len(rollback_reasons) >= 2  # Both conditions required (per plan).

        report = AccuracyReport(
            window=self.window,
            total_decisions=total,
            agreed=stats.get("agreed", 0),
            overridden=stats.get("overridden", 0),
            accuracy=accuracy,
            override_rate=override_rate,
            rollback_needed=rollback_needed,
            rollback_reasons=rollback_reasons,
        )

        if rollback_needed:
            logger.warning(
                "AccuracyTracker: ROLLBACK recommended. Reasons: %s",
                rollback_reasons,
            )
        else:
            logger.debug(
                "AccuracyTracker: accuracy=%.3f override_rate=%.3f (window=%d, total=%d).",
                accuracy, override_rate, self.window, total,
            )

        return report

    def summary_dict(self) -> Dict[str, Any]:
        """Return a serialisable accuracy summary for the monitoring dashboard.

        Returns:
            Dictionary with accuracy metrics and rollback status.
        """
        report = self.evaluate()
        return {
            "window": report.window,
            "total_decisions": report.total_decisions,
            "agreed": report.agreed,
            "overridden": report.overridden,
            "accuracy": report.accuracy,
            "override_rate": report.override_rate,
            "rollback_needed": report.rollback_needed,
            "rollback_reasons": report.rollback_reasons,
            "accuracy_threshold": _settings.rollback_accuracy_threshold,
            "override_rate_threshold": _settings.rollback_override_rate_threshold,
        }
