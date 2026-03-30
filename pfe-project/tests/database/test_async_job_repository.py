"""Tests for persisted async batch job storage."""

from app.database import AsyncBatchJobRepository, get_session_factory, init_database
from app.config import Settings


def test_async_batch_job_repository_persists_job_state(tmp_path) -> None:
    """Ensure async batch jobs can be created and updated through the ORM."""
    database_path = tmp_path / "jobs.db"
    settings = Settings(
        DATABASE_URL=f"sqlite:///{database_path}",
    )
    init_database(settings)
    session_factory = get_session_factory(settings)

    with session_factory() as session:
        repository = AsyncBatchJobRepository(session)
        created = repository.create_job()
        repository.mark_running(created.job_id)
        repository.mark_completed(
            created.job_id,
            {
                "documents": [],
                "metrics": {
                    "document_count": 0,
                    "overall_decisions": {},
                    "field_decisions": {},
                },
                "metadata": {
                    "processed_at": "2026-03-30T00:00:00+00:00",
                    "app_version": "0.1.0",
                    "pipeline_version": "sequential-v1",
                    "extraction_version": "regex-spacy-ensemble-v1",
                    "model_version": "untrained-regex-fallback",
                },
            },
        )

    with session_factory() as session:
        repository = AsyncBatchJobRepository(session)
        loaded = repository.get_job(created.job_id)

    assert loaded is not None
    assert loaded.status == "completed"
    assert loaded.result_payload is not None
    assert loaded.result_payload["metadata"]["model_version"] == "untrained-regex-fallback"
