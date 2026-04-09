"""Repository helpers for database-backed application state."""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from app.database.models import (
    AsyncBatchJob,
    ExtractionResultRecord,
    FieldExtractionRecord,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Serializable data-classes (ORM-free DTOs)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AsyncBatchJobRecord:
    """Serializable async batch job state."""

    job_id: str
    status: str
    submitted_at: str
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    result_payload: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class FieldRecord:
    """Serializable extracted-field record.

    Attributes:
        field_name: NER label (e.g. ``INVOICE_ID``).
        value: Extracted text value.
        confidence: Predicted confidence score.
        decision: Routing decision (``auto`` / ``review`` / ``reject``).
        sources: Extraction sources that contributed.
        scorer: Which scoring mode was active (``heuristic`` or ``rf``).
        char_start: Char start offset in the source text.
        char_end: Char end offset in the source text.
    """

    field_name: str
    value: Optional[str]
    confidence: float
    decision: str
    sources: List[str]
    scorer: str
    char_start: Optional[int] = None
    char_end: Optional[int] = None


@dataclass(frozen=True)
class ExtractionRecord:
    """Serializable extraction result for one document.

    Attributes:
        id: Auto-generated primary key.
        document_id: Unique document identifier.
        source_text: Original input text.
        overall_decision: Document-level routing decision.
        scorer: Scoring mode active for this run.
        model_version: Model version tag used.
        processed_at: ISO-8601 UTC timestamp of processing.
        batch_job_id: Optional parent async-job identifier.
        fields: Resolved field records.
    """

    id: int
    document_id: str
    source_text: str
    overall_decision: str
    scorer: str
    model_version: str
    processed_at: str
    batch_job_id: Optional[str]
    fields: List[FieldRecord]


# ──────────────────────────────────────────────────────────────────────────────
# Async batch job repository (unchanged)
# ──────────────────────────────────────────────────────────────────────────────


class AsyncBatchJobRepository:
    """Persist async batch jobs using SQLAlchemy ORM."""

    def __init__(self, session: Session) -> None:
        """Initialize the repository.

        Args:
            session: Active SQLAlchemy session.
        """
        self.session = session

    def create_job(self) -> AsyncBatchJobRecord:
        """Create and persist a new async job.

        Returns:
            Persisted job record.
        """
        job = AsyncBatchJob(
            job_id=str(uuid4()),
            status="pending",
            submitted_at=self._utc_now(),
        )
        self.session.add(job)
        self.session.commit()
        logger.info("Created persisted async batch job %s.", job.job_id)
        return self._to_record(job)

    def get_job(self, job_id: str) -> Optional[AsyncBatchJobRecord]:
        """Fetch one persisted job.

        Args:
            job_id: Unique job identifier.

        Returns:
            Job record when found, or ``None``.
        """
        job = self.session.get(AsyncBatchJob, job_id)
        return self._to_record(job) if job is not None else None

    def mark_running(self, job_id: str) -> None:
        """Mark a job as running.

        Args:
            job_id: Unique job identifier.
        """
        job = self.session.get(AsyncBatchJob, job_id)
        if job is None:
            return
        job.status = "running"
        self.session.commit()

    def mark_completed(self, job_id: str, result_payload: Dict[str, Any]) -> None:
        """Mark a job as completed.

        Args:
            job_id: Unique job identifier.
            result_payload: Serialized batch result payload.
        """
        job = self.session.get(AsyncBatchJob, job_id)
        if job is None:
            return
        job.status = "completed"
        job.completed_at = self._utc_now()
        job.result_payload = json.dumps(result_payload)
        self.session.commit()
        logger.info("Completed persisted async batch job %s.", job_id)

    def mark_failed(self, job_id: str, error_message: str) -> None:
        """Mark a job as failed.

        Args:
            job_id: Unique job identifier.
            error_message: Failure reason.
        """
        job = self.session.get(AsyncBatchJob, job_id)
        if job is None:
            return
        job.status = "failed"
        job.error_message = error_message
        job.completed_at = self._utc_now()
        self.session.commit()
        logger.error("Persisted async batch job %s failed: %s", job_id, error_message)

    def list_jobs(
        self,
        status: Optional[str] = None,
        limit: int = 20,
    ) -> List[AsyncBatchJobRecord]:
        """List recent async jobs with optional status filtering.

        Args:
            status: Optional job status filter.
            limit: Maximum number of jobs to return.

        Returns:
            Recent job records ordered from newest to oldest.
        """
        statement = (
            select(AsyncBatchJob)
            .order_by(desc(AsyncBatchJob.submitted_at))
            .limit(limit)
        )
        if status is not None:
            statement = statement.where(AsyncBatchJob.status == status)
        jobs = self.session.execute(statement).scalars().all()
        return [self._to_record(job) for job in jobs]

    @staticmethod
    def _to_record(job: AsyncBatchJob) -> AsyncBatchJobRecord:
        """Convert ORM state into a serializable record."""
        payload = json.loads(job.result_payload) if job.result_payload else None
        return AsyncBatchJobRecord(
            job_id=job.job_id,
            status=job.status,
            submitted_at=job.submitted_at,
            completed_at=job.completed_at,
            error_message=job.error_message,
            result_payload=payload,
        )

    @staticmethod
    def _utc_now() -> str:
        """Return an ISO-formatted UTC timestamp."""
        return datetime.now(timezone.utc).isoformat()


# ──────────────────────────────────────────────────────────────────────────────
# Extraction result repository (Month 3)
# ──────────────────────────────────────────────────────────────────────────────


class ExtractionResultRepository:
    """Persist and query pipeline extraction results via SQLAlchemy ORM.

    One ``ExtractionResultRecord`` is written per processed document with
    child ``FieldExtractionRecord`` rows for each resolved field.

    Used by:
    * REST API routes (Month 3) – persist every ``/pipeline/run`` call.
    * KPI engine (Month 3) – query stored results for auto-rate metrics.
    * Monitoring module (Month 4) – drift-detection queries.

    Args:
        session: Active SQLAlchemy ORM session.
    """

    def __init__(self, session: Session) -> None:
        """Initialize the repository.

        Args:
            session: Active SQLAlchemy session.
        """
        self.session = session

    def save(
        self,
        document_id: str,
        source_text: str,
        overall_decision: str,
        scorer: str,
        model_version: str,
        fields: List[FieldRecord],
        batch_job_id: Optional[str] = None,
    ) -> ExtractionRecord:
        """Persist a pipeline result and its child field records.

        Args:
            document_id: Unique document identifier.
            source_text: The original document text.
            overall_decision: Document-level routing decision.
            scorer: Scoring mode (``"heuristic"`` or ``"rf"``).
            model_version: Model version tag used for this run.
            fields: Per-field extraction results.
            batch_job_id: Optional parent async-job identifier.

        Returns:
            The persisted ``ExtractionRecord`` with auto-assigned ``id``.
        """
        orm_result = ExtractionResultRecord(
            document_id=document_id,
            source_text=source_text,
            overall_decision=overall_decision,
            scorer=scorer,
            model_version=model_version,
            processed_at=self._utc_now(),
            batch_job_id=batch_job_id,
        )
        self.session.add(orm_result)
        self.session.flush()  # populate orm_result.id before adding children

        for field in fields:
            orm_field = FieldExtractionRecord(
                extraction_result_id=orm_result.id,
                field_name=field.field_name,
                value=field.value,
                confidence=field.confidence,
                decision=field.decision,
                sources=json.dumps(field.sources),
                scorer=field.scorer,
                char_start=field.char_start,
                char_end=field.char_end,
            )
            self.session.add(orm_field)

        self.session.commit()
        logger.info(
            "Saved extraction result id=%d document_id=%s scorer=%s.",
            orm_result.id,
            document_id,
            scorer,
        )
        # Re-fetch after commit so the relationship loads as a clean list.
        return self.get(orm_result.id)  # type: ignore[return-value]

    def get(self, record_id: int) -> Optional[ExtractionRecord]:
        """Fetch one extraction result by primary key.

        Args:
            record_id: Auto-generated extraction result id.

        Returns:
            The extraction record, or ``None`` if not found.
        """
        row = self.session.get(ExtractionResultRecord, record_id)
        return self._to_extraction_record(row) if row is not None else None

    def list_by_document(
        self,
        document_id: str,
        limit: int = 50,
    ) -> List[ExtractionRecord]:
        """List extraction results for one document, newest first.

        Args:
            document_id: Document identifier to filter by.
            limit: Maximum number of records to return.

        Returns:
            Matching records ordered newest to oldest.
        """
        statement = (
            select(ExtractionResultRecord)
            .where(ExtractionResultRecord.document_id == document_id)
            .order_by(desc(ExtractionResultRecord.processed_at))
            .limit(limit)
        )
        rows = self.session.execute(statement).scalars().all()
        return [self._to_extraction_record(row) for row in rows]

    def list_by_decision(
        self,
        decision: str,
        limit: int = 100,
    ) -> List[ExtractionRecord]:
        """List extraction results filtered by overall decision.

        Args:
            decision: One of ``auto``, ``review``, or ``reject``.
            limit: Maximum number of records to return.

        Returns:
            Matching records ordered newest to oldest.
        """
        statement = (
            select(ExtractionResultRecord)
            .where(ExtractionResultRecord.overall_decision == decision)
            .order_by(desc(ExtractionResultRecord.processed_at))
            .limit(limit)
        )
        rows = self.session.execute(statement).scalars().all()
        return [self._to_extraction_record(row) for row in rows]

    def count_by_decision(self) -> Dict[str, int]:
        """Return document counts grouped by overall decision.

        Returns:
            Mapping of decision label to count, e.g.
            ``{"auto": 40, "review": 10, "reject": 3}``.
        """
        rows = self.session.execute(
            select(
                ExtractionResultRecord.overall_decision,
                func.count(ExtractionResultRecord.id).label("cnt"),
            ).group_by(ExtractionResultRecord.overall_decision)
        ).all()
        return {row.overall_decision: row.cnt for row in rows}

    def average_confidence_by_field(self) -> Dict[str, float]:
        """Return the average confidence per field name across all records.

        Useful for the KPI dashboard confidence-trend chart.

        Returns:
            Mapping of field name to average confidence score.
        """
        rows = self.session.execute(
            select(
                FieldExtractionRecord.field_name,
                func.avg(FieldExtractionRecord.confidence).label("avg_conf"),
            ).group_by(FieldExtractionRecord.field_name)
        ).all()
        return {row.field_name: float(row.avg_conf) for row in rows}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_extraction_record(row: ExtractionResultRecord) -> ExtractionRecord:
        """Convert an ORM row to a serializable ExtractionRecord."""
        field_records: List[FieldRecord] = [
            FieldRecord(
                field_name=f.field_name,
                value=f.value,
                confidence=f.confidence,
                decision=f.decision,
                sources=json.loads(f.sources) if f.sources else [],
                scorer=f.scorer,
                char_start=f.char_start,
                char_end=f.char_end,
            )
            for f in (row.fields or [])
        ]
        return ExtractionRecord(
            id=row.id,
            document_id=row.document_id,
            source_text=row.source_text,
            overall_decision=row.overall_decision,
            scorer=row.scorer,
            model_version=row.model_version,
            processed_at=row.processed_at,
            batch_job_id=row.batch_job_id,
            fields=field_records,
        )

    @staticmethod
    def _utc_now() -> str:
        """Return an ISO-formatted UTC timestamp."""
        return datetime.now(timezone.utc).isoformat()
