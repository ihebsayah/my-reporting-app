"""In-memory job storage for async batch API workflows."""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Dict, Optional
from uuid import uuid4

from app.pipeline.batch_processor import BatchProcessingResult

logger = logging.getLogger(__name__)


@dataclass
class BatchJobRecord:
    """Represent one async batch pipeline job."""

    job_id: str
    status: str
    submitted_at: str
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    result: Optional[BatchProcessingResult] = None


class InMemoryBatchJobStore:
    """Store async batch jobs in memory for local and test environments."""

    def __init__(self) -> None:
        """Initialize the in-memory store."""
        self._jobs: Dict[str, BatchJobRecord] = {}
        self._lock = Lock()

    def create_job(self) -> BatchJobRecord:
        """Create and store a new pending job.

        Returns:
            Newly created job record.
        """
        job_id = str(uuid4())
        record = BatchJobRecord(
            job_id=job_id,
            status="pending",
            submitted_at=self._utc_now(),
        )
        with self._lock:
            self._jobs[job_id] = record
        logger.info("Created async batch job %s.", job_id)
        return record

    def get_job(self, job_id: str) -> Optional[BatchJobRecord]:
        """Return a stored job by ID.

        Args:
            job_id: Unique job identifier.

        Returns:
            Matching job record when found.
        """
        with self._lock:
            return self._jobs.get(job_id)

    def mark_running(self, job_id: str) -> None:
        """Mark a job as running.

        Args:
            job_id: Unique job identifier.
        """
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].status = "running"

    def mark_completed(self, job_id: str, result: BatchProcessingResult) -> None:
        """Mark a job as completed and attach its result.

        Args:
            job_id: Unique job identifier.
            result: Batch pipeline result.
        """
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].status = "completed"
                self._jobs[job_id].result = result
                self._jobs[job_id].completed_at = self._utc_now()
        logger.info("Completed async batch job %s.", job_id)

    def mark_failed(self, job_id: str, error_message: str) -> None:
        """Mark a job as failed.

        Args:
            job_id: Unique job identifier.
            error_message: Failure reason.
        """
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].status = "failed"
                self._jobs[job_id].error_message = error_message
                self._jobs[job_id].completed_at = self._utc_now()
        logger.error("Async batch job %s failed: %s", job_id, error_message)

    @staticmethod
    def _utc_now() -> str:
        """Return an ISO-formatted UTC timestamp."""
        return datetime.now(timezone.utc).isoformat()
