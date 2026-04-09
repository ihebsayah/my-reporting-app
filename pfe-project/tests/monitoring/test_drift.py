"""Tests for the confidence drift detector."""

import pytest

from app.monitoring.drift import (
    ConfidenceDriftDetector,
    DriftReport,
    ExtractionSample,
    WindowStats,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _samples(n: int, decision: str = "auto", confidence: float = 0.92) -> list:
    """Generate n identical ExtractionSample objects."""
    return [
        ExtractionSample(
            overall_decision=decision,
            mean_field_confidence=confidence,
            processed_at=f"2026-04-{i+1:02d}T10:00:00Z",
        )
        for i in range(n)
    ]


def _mixed_samples(baseline_n: int, recent_n: int, recent_conf: float, recent_decision: str) -> list:
    """Stable baseline samples followed by degraded recent samples."""
    baseline = _samples(baseline_n, "auto", 0.92)
    recent = _samples(recent_n, recent_decision, recent_conf)
    # Tag timestamps so they sort correctly after baseline.
    for i, s in enumerate(recent):
        object.__setattr__(s, "processed_at", f"2026-05-{i+1:02d}T10:00:00Z")
    return baseline + recent


# ──────────────────────────────────────────────────────────────────────────────
# WindowStats calculation
# ──────────────────────────────────────────────────────────────────────────────


def test_compute_window_stats_all_auto() -> None:
    """All-auto window must give auto_rate=1.0 and zero other rates."""
    samples = _samples(10, "auto", 0.95)
    stats = ConfidenceDriftDetector._compute_window_stats(samples)
    assert stats.auto_rate == pytest.approx(1.0)
    assert stats.review_rate == pytest.approx(0.0)
    assert stats.reject_rate == pytest.approx(0.0)
    assert stats.mean_confidence == pytest.approx(0.95, abs=1e-3)


def test_compute_window_stats_mixed_decisions() -> None:
    """Mixed decisions must produce correct fractional rates."""
    samples = (
        _samples(8, "auto", 0.95) +
        _samples(1, "review", 0.80) +
        _samples(1, "reject", 0.50)
    )
    stats = ConfidenceDriftDetector._compute_window_stats(samples)
    assert stats.window_size == 10
    assert stats.auto_rate == pytest.approx(0.8)
    assert stats.review_rate == pytest.approx(0.1)
    assert stats.reject_rate == pytest.approx(0.1)


def test_compute_window_stats_empty() -> None:
    """Empty sample list must return a zero-valued WindowStats."""
    stats = ConfidenceDriftDetector._compute_window_stats([])
    assert stats.window_size == 0
    assert stats.auto_rate == pytest.approx(0.0)
    assert stats.mean_confidence == pytest.approx(0.0)


# ──────────────────────────────────────────────────────────────────────────────
# detect() — insufficient samples
# ──────────────────────────────────────────────────────────────────────────────


def test_detect_returns_no_drift_when_insufficient_samples() -> None:
    """detect() must return drift_detected=False when samples < baseline + recent."""
    detector = ConfidenceDriftDetector(baseline_window_size=50, recent_window_size=20)
    report = detector.detect(_samples(30))  # only 30, need 70

    assert isinstance(report, DriftReport)
    assert report.drift_detected is False
    assert report.triggered_signals == []


# ──────────────────────────────────────────────────────────────────────────────
# detect() — no drift
# ──────────────────────────────────────────────────────────────────────────────


def test_detect_no_drift_with_stable_samples() -> None:
    """Stable samples with identical baseline and recent must show no drift."""
    detector = ConfidenceDriftDetector(baseline_window_size=10, recent_window_size=5)
    samples = _samples(20, "auto", 0.92)
    report = detector.detect(samples)

    assert report.drift_detected is False
    assert report.auto_rate_drop == pytest.approx(0.0, abs=0.01)
    assert report.confidence_drop == pytest.approx(0.0, abs=0.01)


# ──────────────────────────────────────────────────────────────────────────────
# detect() — auto-rate drift
# ──────────────────────────────────────────────────────────────────────────────


def test_detect_auto_rate_drift_triggers() -> None:
    """A 20pp auto-rate drop in the recent window must trigger auto_rate drift."""
    detector = ConfidenceDriftDetector(
        auto_rate_drop_threshold=0.10,
        confidence_drop_threshold=0.05,
        baseline_window_size=10,
        recent_window_size=10,
    )
    # Baseline: all auto.  Recent: all review.
    samples = _samples(10, "auto", 0.92) + _samples(10, "review", 0.75)
    for i, s in enumerate(samples[10:], 11):
        object.__setattr__(s, "processed_at", f"2026-05-{i:02d}T10:00:00Z")
    report = detector.detect(samples)

    assert report.drift_detected is True
    assert "auto_rate" in report.triggered_signals
    assert report.auto_rate_drop == pytest.approx(1.0, abs=0.01)


# ──────────────────────────────────────────────────────────────────────────────
# detect() — confidence drift
# ──────────────────────────────────────────────────────────────────────────────


def test_detect_confidence_drift_triggers() -> None:
    """A confidence drop > threshold must trigger the confidence drift signal."""
    detector = ConfidenceDriftDetector(
        auto_rate_drop_threshold=0.10,
        confidence_drop_threshold=0.05,
        baseline_window_size=10,
        recent_window_size=10,
    )
    # Baseline: 0.92 confidence.  Recent: 0.80 confidence (12pp drop > 5pp).
    baseline = _samples(10, "auto", 0.92)
    recent = [
        ExtractionSample("auto", 0.80, f"2026-05-{i+1:02d}T10:00:00Z")
        for i in range(10)
    ]
    report = detector.detect(baseline + recent)

    assert report.drift_detected is True
    assert "confidence" in report.triggered_signals
    assert report.confidence_drop == pytest.approx(0.12, abs=0.01)


# ──────────────────────────────────────────────────────────────────────────────
# detect() — both signals
# ──────────────────────────────────────────────────────────────────────────────


def test_detect_both_signals_can_trigger_simultaneously() -> None:
    """Both auto_rate and confidence signals may appear in triggered_signals."""
    detector = ConfidenceDriftDetector(
        auto_rate_drop_threshold=0.10,
        confidence_drop_threshold=0.05,
        baseline_window_size=10,
        recent_window_size=10,
    )
    baseline = _samples(10, "auto", 0.92)
    recent = [
        ExtractionSample("reject", 0.40, f"2026-05-{i+1:02d}T10:00:00Z")
        for i in range(10)
    ]
    report = detector.detect(baseline + recent)

    assert report.drift_detected is True
    assert "auto_rate" in report.triggered_signals
    assert "confidence" in report.triggered_signals


# ──────────────────────────────────────────────────────────────────────────────
# DriftReport fields
# ──────────────────────────────────────────────────────────────────────────────


def test_drift_report_contains_checked_at_timestamp() -> None:
    """DriftReport must include a non-empty checked_at ISO timestamp."""
    detector = ConfidenceDriftDetector(baseline_window_size=5, recent_window_size=5)
    report = detector.detect(_samples(15, "auto", 0.92))
    assert report.checked_at
    assert "T" in report.checked_at  # ISO-8601 with time component


def test_drift_report_thresholds_match_detector_config() -> None:
    """DriftReport must echo the configured thresholds."""
    detector = ConfidenceDriftDetector(
        auto_rate_drop_threshold=0.15,
        confidence_drop_threshold=0.08,
        baseline_window_size=5,
        recent_window_size=5,
    )
    report = detector.detect(_samples(15))
    assert report.auto_rate_threshold == pytest.approx(0.15)
    assert report.confidence_threshold == pytest.approx(0.08)


# ──────────────────────────────────────────────────────────────────────────────
# samples_from_records() helper
# ──────────────────────────────────────────────────────────────────────────────


def test_samples_from_records_sorts_oldest_first() -> None:
    """samples_from_records must sort by processed_at, oldest first."""
    from dataclasses import dataclass

    @dataclass
    class FakeField:
        confidence: float

    @dataclass
    class FakeRecord:
        overall_decision: str
        processed_at: str
        fields: list

    records = [
        FakeRecord("auto", "2026-04-03T10:00:00Z", [FakeField(0.9)]),
        FakeRecord("review", "2026-04-01T10:00:00Z", [FakeField(0.75)]),
        FakeRecord("auto", "2026-04-02T10:00:00Z", [FakeField(0.92)]),
    ]
    detector = ConfidenceDriftDetector()
    samples = detector.samples_from_records(records)

    timestamps = [s.processed_at for s in samples]
    assert timestamps == sorted(timestamps)
    assert samples[0].overall_decision == "review"  # oldest → 2026-04-01
