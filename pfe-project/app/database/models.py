"""Database ORM models."""

from typing import List

from sqlalchemy import Float, ForeignKey, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

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


class ExtractionResultRecord(Base):
    """Persisted pipeline run result for one document.

    One row per document processed by ``SequentialExtractionDecisionEngine``.
    Child rows are stored in ``FieldExtractionRecord``.
    """

    __tablename__ = "extraction_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    source_text: Mapped[str] = mapped_column(Text, nullable=False)
    overall_decision: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    scorer: Mapped[str] = mapped_column(Text, nullable=False, default="heuristic")
    model_version: Mapped[str] = mapped_column(Text, nullable=False, default="unknown")
    processed_at: Mapped[str] = mapped_column(Text, nullable=False)
    batch_job_id: Mapped[str] = mapped_column(Text, nullable=True, index=True)

    fields: Mapped[List["FieldExtractionRecord"]] = relationship(
        "FieldExtractionRecord",
        back_populates="extraction_result",
        cascade="all, delete-orphan",
        lazy="select",
        uselist=True,
    )


class FieldExtractionRecord(Base):
    """One extracted field within a persisted pipeline result.

    Child of ``ExtractionResultRecord``.  Stores the entity text, confidence,
    decision, and sources for every field the pipeline resolved.
    """

    __tablename__ = "field_extraction_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    extraction_result_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("extraction_results.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    field_name: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    value: Mapped[str] = mapped_column(Text, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    decision: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    sources: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    scorer: Mapped[str] = mapped_column(Text, nullable=False, default="heuristic")
    char_start: Mapped[int] = mapped_column(Integer, nullable=True)
    char_end: Mapped[int] = mapped_column(Integer, nullable=True)

    extraction_result: Mapped[ExtractionResultRecord] = relationship(
        "ExtractionResultRecord",
        back_populates="fields",
    )
