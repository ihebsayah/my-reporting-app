"""Repository helpers for database-backed application state."""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import uuid4

from sqlalchemy.orm import Session

from app.database.models import AsyncBatchJob

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AsyncBatchJobRecord:
    """Serializable async batch job state."""

    job_id: str
    status: str
    submitted_at: str
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    result_payload: Optional[Dict[str, Any]] = None


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
            Job record when found.
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
