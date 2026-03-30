"""Database ORM models."""

from sqlalchemy import Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base


class AsyncBatchJob(Base):
    """Persisted async batch pipeline job."""

    __tablename__ = "async_batch_jobs"

    job_id: Mapped[str] = mapped_column(Text, primary_key=True)
    status: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    submitted_at: Mapped[str] = mapped_column(Text, nullable=False)
    completed_at: Mapped[str] = mapped_column(Text, nullable=True)
    error_message: Mapped[str] = mapped_column(Text, nullable=True)
    result_payload: Mapped[str] = mapped_column(Text, nullable=True)
