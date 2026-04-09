"""Dashboard data services — Month 4 final version.

Adds four new methods to ``DashboardDataService``:

1. ``build_storage_kpi()`` — auto/review/reject rates from persisted DB records.
2. ``build_drift_report()`` — sliding-window drift check (needs ≥70 DB records).
3. ``trigger_retraining()`` — runs ``RFRetrainingPipeline`` and hot-swaps engine.
4. ``get_extraction_history()`` — historical runs for one document from DB.

The original ``build_dashboard_data()`` and ``load_document_preview()`` are
unchanged so existing dashboard tests continue to pass.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import Settings, get_settings
from app.database import (
    AsyncBatchJobRepository,
    ExtractionResultRepository,
    get_session_factory,
    init_database,
)
from app.kpi import ExtractionKPIService, PipelineKPIService
from app.kpi.metrics import ExtractionStorageKPI
from app.monitoring.drift import ConfidenceDriftDetector, DriftReport
from app.pipeline.batch_processor import PipelineBatchProcessor
from app.pipeline.decision_engine import RFModelLoader, SequentialExtractionDecisionEngine

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data-classes (unchanged from earlier months)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DashboardMetricCard:
    """Represent one dashboard metric card."""

    label: str
    value: str


@dataclass(frozen=True)
class DashboardJobSummary:
    """Represent one recent async job summary."""

    job_id: str
    status: str
    submitted_at: str
    completed_at: Optional[str]
    error_message: Optional[str]


@dataclass(frozen=True)
class DashboardFieldKPI:
    """Represent one field KPI row for the dashboard."""

    field_name: str
    total_occurrences: int
    auto_count: int
    review_count: int
    reject_count: int
    average_confidence: float


@dataclass(frozen=True)
class DashboardData:
    """Represent all data needed by the dashboard."""

    metric_cards: List[DashboardMetricCard] = field(default_factory=list)
    field_kpis: List[DashboardFieldKPI] = field(default_factory=list)
    recent_jobs: List[DashboardJobSummary] = field(default_factory=list)
    field_decisions: Dict[str, Dict[str, int]] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# Dashboard service
# ──────────────────────────────────────────────────────────────────────────────


class DashboardDataService:
    """Build dashboard data from pipeline and database services.

    Args:
        settings: Optional settings override (useful in tests).
        batch_processor: Optional batch processor override.
        kpi_service: Optional KPI service override.
        engine: Optional live pipeline engine (used for hot-swap after retrain).
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        batch_processor: Optional[PipelineBatchProcessor] = None,
        kpi_service: Optional[PipelineKPIService] = None,
        engine: Optional[SequentialExtractionDecisionEngine] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.batch_processor = batch_processor or PipelineBatchProcessor(settings=self.settings)
        self.kpi_service = kpi_service or PipelineKPIService()
        self.engine = engine
        self.session_factory = get_session_factory(self.settings)
        init_database(self.settings)

    # ── Original Month 1-2 methods (unchanged) ─────────────────────────────

    def build_dashboard_data(
        self,
        input_dir: str = "docs/source_documents",
        job_limit: int = 10,
    ) -> DashboardData:
        """Build dashboard data from source documents and persisted jobs.

        Args:
            input_dir: Source document directory.
            job_limit: Maximum number of jobs to display.

        Returns:
            Structured dashboard data.
        """
        batch_result = self.batch_processor.run_directory(input_dir)
        report = self.kpi_service.build_report(batch_result)
        recent_jobs = self._list_recent_jobs(limit=job_limit)
        metric_cards = [
            DashboardMetricCard(label="Documents", value=str(report.document_count)),
            DashboardMetricCard(label="Auto ✅", value=str(report.auto_documents)),
            DashboardMetricCard(label="Review 🔍", value=str(report.review_documents)),
            DashboardMetricCard(label="Reject ❌", value=str(report.reject_documents)),
            DashboardMetricCard(
                label="Avg Confidence",
                value=f"{report.average_field_confidence:.2f}",
            ),
        ]
        return DashboardData(
            metric_cards=metric_cards,
            field_kpis=[
                DashboardFieldKPI(
                    field_name=item.field_name,
                    total_occurrences=item.total_occurrences,
                    auto_count=item.auto_count,
                    review_count=item.review_count,
                    reject_count=item.reject_count,
                    average_confidence=item.average_confidence,
                )
                for item in report.field_kpis
            ],
            recent_jobs=recent_jobs,
            field_decisions=batch_result.metrics.field_decisions if batch_result.metrics else {},
        )

    def load_document_preview(self, input_dir: str = "docs/source_documents") -> List[Dict[str, str]]:
        """Load source documents for a lightweight dashboard preview.

        Args:
            input_dir: Source document directory.

        Returns:
            Preview rows with document ID, decision, and truncated text.
        """
        batch_documents = self.batch_processor.run_directory(input_dir).documents
        return [
            {
                "document_id": item.document_id,
                "preview": item.result.text.strip().replace("\n", " ")[:120],
                "overall_decision": item.result.overall_decision,
            }
            for item in batch_documents
        ]

    # ── Month 3: storage KPI ────────────────────────────────────────────────

    def build_storage_kpi(self) -> ExtractionStorageKPI:
        """Build auto/review/reject rates from persisted extraction DB records.

        Returns:
            ``ExtractionStorageKPI`` with rates and per-field confidence averages.
        """
        with self.session_factory() as session:
            repo = ExtractionResultRepository(session)
            counts = repo.count_by_decision()
            avg_conf = repo.average_confidence_by_field()
        return ExtractionKPIService().from_aggregates(
            decision_counts=counts,
            avg_confidence_by_field=avg_conf,
        )

    # ── Month 4: drift detection ────────────────────────────────────────────

    def build_drift_report(
        self,
        baseline_window: int = 50,
        recent_window: int = 20,
        auto_rate_threshold: float = 0.10,
        confidence_threshold: float = 0.05,
    ) -> Optional[DriftReport]:
        """Run drift detection over persisted extraction records.

        Returns ``None`` if fewer than ``baseline_window + recent_window``
        records exist (insufficient history for a meaningful comparison).

        Args:
            baseline_window: Number of records for the baseline window.
            recent_window: Number of records for the recent window.
            auto_rate_threshold: Auto-rate drop that triggers drift.
            confidence_threshold: Confidence drop that triggers drift.

        Returns:
            A ``DriftReport``, or ``None`` if there is insufficient history.
        """
        needed = baseline_window + recent_window
        with self.session_factory() as session:
            # Fetch newest `needed` records (search is newest-first from DB).
            records = ExtractionResultRepository(session).list_by_decision(
                decision="auto", limit=needed
            ) + ExtractionResultRepository(session).list_by_decision(
                decision="review", limit=needed
            ) + ExtractionResultRepository(session).list_by_decision(
                decision="reject", limit=needed
            )

        detector = ConfidenceDriftDetector(
            auto_rate_drop_threshold=auto_rate_threshold,
            confidence_drop_threshold=confidence_threshold,
            baseline_window_size=baseline_window,
            recent_window_size=recent_window,
        )
        samples = detector.samples_from_records(records)

        if len(samples) < needed:
            logger.info(
                "Drift check skipped: only %d records (need %d).", len(samples), needed
            )
            return None

        return detector.detect(samples)

    # ── Month 4: retraining trigger ─────────────────────────────────────────

    def trigger_retraining(
        self,
        base_jsonl_path: Optional[Path] = None,
        n_estimators: int = 200,
    ) -> Dict[str, Any]:
        """Run the monthly RF retraining pipeline and hot-swap the engine.

        Args:
            base_jsonl_path: Override path to base training JSONL.  Defaults
                to ``docs/annotation/rf_training_records.jsonl``.
            n_estimators: Number of RF trees.

        Returns:
            A dict with ``success``, ``model_version``, ``accuracy``,
            ``total_records``, or ``error`` on failure.
        """
        from app.retraining.pipeline import RFRetrainingPipeline

        resolved_base = base_jsonl_path or Path(
            "docs/annotation/rf_training_records.jsonl"
        )
        try:
            pipeline = RFRetrainingPipeline(settings=self.settings, use_bert=False)
            result = pipeline.run(
                base_jsonl_path=resolved_base,
                output_dir=Path(self.settings.rf_model_output_dir),
                n_estimators=n_estimators,
                engine=self.engine,
            )
            logger.info("Retraining triggered via dashboard: %s.", result.training_result.model_version)
            return {
                "success": True,
                "model_version": result.training_result.model_version,
                "accuracy": result.training_result.accuracy,
                "total_records": result.total_records,
                "feedback_records": result.feedback_records,
                "model_path": result.model_path,
            }
        except Exception as exc:
            logger.error("Dashboard retraining failed: %s.", exc)
            return {"success": False, "error": str(exc)}

    # ── Month 4: extraction history ─────────────────────────────────────────

    def get_extraction_history(
        self,
        document_id: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Fetch historical extraction results for one document.

        Args:
            document_id: Document identifier to query.
            limit: Maximum number of records to return.

        Returns:
            List of serializable extraction history dicts.
        """
        with self.session_factory() as session:
            records = ExtractionResultRepository(session).list_by_document(
                document_id, limit=limit
            )
        return [
            {
                "record_id": record.id,
                "document_id": record.document_id,
                "overall_decision": record.overall_decision,
                "scorer": record.scorer,
                "model_version": record.model_version,
                "processed_at": record.processed_at,
                "fields": [
                    {
                        "field_name": f.field_name,
                        "value": f.value,
                        "confidence": f.confidence,
                        "decision": f.decision,
                        "sources": f.sources,
                        "scorer": f.scorer,
                    }
                    for f in record.fields
                ],
            }
            for record in records
        ]

    # ── Private helpers ─────────────────────────────────────────────────────

    def _list_recent_jobs(self, limit: int) -> List[DashboardJobSummary]:
        """Fetch recent persisted jobs for the dashboard."""
        with self.session_factory() as session:
            jobs = AsyncBatchJobRepository(session).list_jobs(limit=limit)
        return [
            DashboardJobSummary(
                job_id=job.job_id,
                status=job.status,
                submitted_at=job.submitted_at,
                completed_at=job.completed_at,
                error_message=job.error_message,
            )
            for job in jobs
        ]


def dashboard_source_exists(input_dir: str) -> bool:
    """Return whether the dashboard source directory exists.

    Args:
        input_dir: Source document directory.

    Returns:
        True when the source directory exists.
    """
    return Path(input_dir).exists()
