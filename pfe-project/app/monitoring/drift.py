"""Confidence drift detector for the extraction pipeline.

Month 4 monitoring module.

Drift is detected by comparing a rolling window of recent pipeline decisions
(queried from the DB) against a stable baseline window.  Two signals are
monitored:

1. **Auto-rate drift** — the fraction of documents routed to ``auto``
   drops by more than ``auto_rate_drop_threshold`` vs the baseline.
2. **Confidence drift** — the mean confidence over the window falls by more
   than ``confidence_drop_threshold`` vs the baseline.

These thresholds deliberately mirror the architecture's target metrics
(≥85% auto-rate, ≥0.90 avg confidence) so the monitoring system is
tightly coupled to project KPIs.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data types
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class WindowStats:
    """Aggregate stats over one sliding window of extraction records.

    Attributes:
        window_size: Number of records in this window.
        auto_rate: Fraction of records with ``overall_decision == "auto"``.
        review_rate: Fraction with ``overall_decision == "review"``.
        reject_rate: Fraction with ``overall_decision == "reject"``.
        mean_confidence: Average field confidence across all records.
    """

    window_size: int
    auto_rate: float
    review_rate: float
    reject_rate: float
    mean_confidence: float


@dataclass(frozen=True)
class DriftReport:
    """Result of a drift-detection comparison.

    Attributes:
        baseline: Stats computed from the baseline window.
        recent: Stats computed from the recent window.
        auto_rate_drop: Difference (baseline - recent) in auto-rate.
        confidence_drop: Difference (baseline - recent) in mean confidence.
        drift_detected: True when either signal exceeds its threshold.
        triggered_signals: Which signals triggered (e.g. ``["auto_rate"]``).
        auto_rate_threshold: The configured auto-rate drop threshold.
        confidence_threshold: The configured confidence drop threshold.
        checked_at: ISO-8601 UTC timestamp of the check.
    """

    baseline: WindowStats
    recent: WindowStats
    auto_rate_drop: float
    confidence_drop: float
    drift_detected: bool
    triggered_signals: List[str]
    auto_rate_threshold: float
    confidence_threshold: float
    checked_at: str


# ──────────────────────────────────────────────────────────────────────────────
# Sample record type (DB-agnostic)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ExtractionSample:
    """A minimal extraction sample used by the drift detector.

    Decoupled from ORM so the detector can work with any data source
    (DB rows, CSV exports, test fixtures).

    Attributes:
        overall_decision: ``auto``, ``review``, or ``reject``.
        mean_field_confidence: Average confidence across extracted fields.
        processed_at: ISO-8601 UTC timestamp.
    """

    overall_decision: str
    mean_field_confidence: float
    processed_at: str


# ──────────────────────────────────────────────────────────────────────────────
# Drift detector
# ──────────────────────────────────────────────────────────────────────────────


class ConfidenceDriftDetector:
    """Detect confidence and auto-rate drift over sliding windows.

    Splits ``samples`` into:
    * **Baseline window** — the oldest ``baseline_window_size`` samples.
    * **Recent window** — the newest ``recent_window_size`` samples.

    If the recent window shows a significant drop in either signal vs the
    baseline, a ``DriftReport`` with ``drift_detected=True`` is returned.

    Args:
        auto_rate_drop_threshold: Minimum auto-rate drop to trigger drift.
            Default 0.10 (10 pp drop).
        confidence_drop_threshold: Minimum mean-confidence drop to trigger.
            Default 0.05 (5 pp drop in [0, 1] scale).
        baseline_window_size: Number of samples in the baseline window.
        recent_window_size: Number of samples in the recent window.

    Example::

        detector = ConfidenceDriftDetector()
        report = detector.detect(samples)
        if report.drift_detected:
            logger.warning("Drift detected: %s", report.triggered_signals)
    """

    def __init__(
        self,
        auto_rate_drop_threshold: float = 0.10,
        confidence_drop_threshold: float = 0.05,
        baseline_window_size: int = 50,
        recent_window_size: int = 20,
    ) -> None:
        """Initialize the drift detector.

        Args:
            auto_rate_drop_threshold: Auto-rate drop threshold.
            confidence_drop_threshold: Confidence drop threshold.
            baseline_window_size: Samples in the baseline window.
            recent_window_size: Samples in the recent window.
        """
        self.auto_rate_drop_threshold = auto_rate_drop_threshold
        self.confidence_drop_threshold = confidence_drop_threshold
        self.baseline_window_size = baseline_window_size
        self.recent_window_size = recent_window_size

    def detect(self, samples: Sequence[ExtractionSample]) -> DriftReport:
        """Run drift detection over a sequence of extraction samples.

        Samples should be ordered from oldest to newest.

        Args:
            samples: Ordered extraction samples (oldest first).

        Returns:
            A ``DriftReport`` describing the comparison result.
        """
        total = len(samples)
        checked_at = datetime.now(timezone.utc).isoformat()

        if total < self.baseline_window_size + self.recent_window_size:
            logger.info(
                "Insufficient samples for drift detection (%d < %d required).",
                total,
                self.baseline_window_size + self.recent_window_size,
            )
            empty = WindowStats(
                window_size=0,
                auto_rate=0.0,
                review_rate=0.0,
                reject_rate=0.0,
                mean_confidence=0.0,
            )
            return DriftReport(
                baseline=empty,
                recent=empty,
                auto_rate_drop=0.0,
                confidence_drop=0.0,
                drift_detected=False,
                triggered_signals=[],
                auto_rate_threshold=self.auto_rate_drop_threshold,
                confidence_threshold=self.confidence_drop_threshold,
                checked_at=checked_at,
            )

        baseline_samples = list(samples[: self.baseline_window_size])
        recent_samples = list(samples[-self.recent_window_size :])

        baseline_stats = self._compute_window_stats(baseline_samples)
        recent_stats = self._compute_window_stats(recent_samples)

        auto_rate_drop = round(baseline_stats.auto_rate - recent_stats.auto_rate, 4)
        confidence_drop = round(baseline_stats.mean_confidence - recent_stats.mean_confidence, 4)

        triggered: List[str] = []
        if auto_rate_drop >= self.auto_rate_drop_threshold:
            triggered.append("auto_rate")
        if confidence_drop >= self.confidence_drop_threshold:
            triggered.append("confidence")

        drift_detected = len(triggered) > 0

        if drift_detected:
            logger.warning(
                "Drift detected: signals=%s auto_rate_drop=%.3f confidence_drop=%.3f.",
                triggered,
                auto_rate_drop,
                confidence_drop,
            )
        else:
            logger.info(
                "No drift detected: auto_rate_drop=%.3f confidence_drop=%.3f.",
                auto_rate_drop,
                confidence_drop,
            )

        return DriftReport(
            baseline=baseline_stats,
            recent=recent_stats,
            auto_rate_drop=auto_rate_drop,
            confidence_drop=confidence_drop,
            drift_detected=drift_detected,
            triggered_signals=triggered,
            auto_rate_threshold=self.auto_rate_drop_threshold,
            confidence_threshold=self.confidence_drop_threshold,
            checked_at=checked_at,
        )

    @staticmethod
    def _compute_window_stats(samples: List[ExtractionSample]) -> WindowStats:
        """Compute aggregate stats for one window of samples."""
        n = len(samples)
        if n == 0:
            return WindowStats(
                window_size=0,
                auto_rate=0.0,
                review_rate=0.0,
                reject_rate=0.0,
                mean_confidence=0.0,
            )
        auto = sum(1 for s in samples if s.overall_decision == "auto")
        review = sum(1 for s in samples if s.overall_decision == "review")
        reject = sum(1 for s in samples if s.overall_decision == "reject")
        mean_conf = sum(s.mean_field_confidence for s in samples) / n
        return WindowStats(
            window_size=n,
            auto_rate=round(auto / n, 4),
            review_rate=round(review / n, 4),
            reject_rate=round(reject / n, 4),
            mean_confidence=round(mean_conf, 4),
        )

    def samples_from_records(
        self,
        records: Sequence,  # Sequence[ExtractionRecord] from DB repository
    ) -> List[ExtractionSample]:
        """Convert DB ``ExtractionRecord`` objects into ``ExtractionSample`` objects.

        Args:
            records: ExtractionRecord rows from ``ExtractionResultRepository``.

        Returns:
            Ordered list of ``ExtractionSample`` objects (oldest first).
        """
        samples: List[ExtractionSample] = []
        for record in records:
            field_confs = [f.confidence for f in record.fields]
            mean_conf = sum(field_confs) / len(field_confs) if field_confs else 0.0
            samples.append(
                ExtractionSample(
                    overall_decision=record.overall_decision,
                    mean_field_confidence=mean_conf,
                    processed_at=record.processed_at,
                )
            )
        # Sort oldest-first so sliding window math is correct.
        return sorted(samples, key=lambda s: s.processed_at)
