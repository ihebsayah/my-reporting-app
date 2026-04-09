"""Tests for ExtractionResultRepository (Month 3)."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database.base import Base
from app.database.models import ExtractionResultRecord, FieldExtractionRecord
from app.database.repositories import (
    ExtractionRecord,
    ExtractionResultRepository,
    FieldRecord,
)


# ──────────────────────────────────────────────────────────────────────────────
# In-memory SQLite test database
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def session():
    """Provide a fresh in-memory SQLite session per test."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    with factory() as s:
        yield s
    Base.metadata.drop_all(engine)


def _sample_fields() -> list:
    return [
        FieldRecord(
            field_name="INVOICE_ID",
            value="INV-001",
            confidence=0.95,
            decision="auto",
            sources=["regex"],
            scorer="heuristic",
            char_start=8,
            char_end=15,
        ),
        FieldRecord(
            field_name="TOTAL_AMOUNT",
            value="$1,200.00",
            confidence=0.72,
            decision="review",
            sources=["regex", "spacy"],
            scorer="heuristic",
            char_start=64,
            char_end=73,
        ),
    ]


# ──────────────────────────────────────────────────────────────────────────────
# save()
# ──────────────────────────────────────────────────────────────────────────────


def test_save_returns_extraction_record(session) -> None:
    """save() must return an ExtractionRecord with a valid id."""
    repo = ExtractionResultRepository(session)
    record = repo.save(
        document_id="doc-001",
        source_text="Invoice INV-001 total $1,200.00",
        overall_decision="review",
        scorer="heuristic",
        model_version="untrained-regex-fallback",
        fields=_sample_fields(),
    )
    assert isinstance(record, ExtractionRecord)
    assert record.id > 0
    assert record.document_id == "doc-001"
    assert record.overall_decision == "review"


def test_save_persists_child_field_records(session) -> None:
    """save() must write child FieldExtractionRecord rows to the DB."""
    repo = ExtractionResultRepository(session)
    record = repo.save(
        document_id="doc-002",
        source_text="some text",
        overall_decision="auto",
        scorer="rf",
        model_version="rf_confidence_v20260408_1500",
        fields=_sample_fields(),
    )
    assert len(record.fields) == 2
    labels = {f.field_name for f in record.fields}
    assert labels == {"INVOICE_ID", "TOTAL_AMOUNT"}


def test_save_stores_sources_as_list(session) -> None:
    """save() must round-trip the sources list correctly."""
    repo = ExtractionResultRepository(session)
    record = repo.save(
        document_id="doc-003",
        source_text="text",
        overall_decision="auto",
        scorer="heuristic",
        model_version="v1",
        fields=[
            FieldRecord(
                field_name="INVOICE_ID",
                value="INV-X",
                confidence=0.9,
                decision="auto",
                sources=["regex", "spacy"],
                scorer="heuristic",
            )
        ],
    )
    assert record.fields[0].sources == ["regex", "spacy"]


def test_save_allows_null_batch_job_id(session) -> None:
    """batch_job_id defaults to None when not provided."""
    repo = ExtractionResultRepository(session)
    record = repo.save(
        document_id="doc-004",
        source_text="x",
        overall_decision="auto",
        scorer="heuristic",
        model_version="v1",
        fields=_sample_fields(),
    )
    assert record.batch_job_id is None


# ──────────────────────────────────────────────────────────────────────────────
# get()
# ──────────────────────────────────────────────────────────────────────────────


def test_get_returns_saved_record(session) -> None:
    """get() must retrieve the record by primary key."""
    repo = ExtractionResultRepository(session)
    saved = repo.save(
        document_id="doc-005",
        source_text="invoice text",
        overall_decision="auto",
        scorer="rf",
        model_version="rf_v1",
        fields=_sample_fields(),
    )
    fetched = repo.get(saved.id)
    assert fetched is not None
    assert fetched.id == saved.id
    assert fetched.document_id == "doc-005"


def test_get_returns_none_for_missing_id(session) -> None:
    """get() must return None when the id does not exist."""
    repo = ExtractionResultRepository(session)
    assert repo.get(99999) is None


# ──────────────────────────────────────────────────────────────────────────────
# list_by_document()
# ──────────────────────────────────────────────────────────────────────────────


def test_list_by_document_returns_all_records_for_document(session) -> None:
    """list_by_document() must return every record for a given document_id."""
    repo = ExtractionResultRepository(session)
    for _ in range(3):
        repo.save(
            document_id="doc-repeat",
            source_text="text",
            overall_decision="auto",
            scorer="heuristic",
            model_version="v1",
            fields=_sample_fields(),
        )
    results = repo.list_by_document("doc-repeat")
    assert len(results) == 3
    assert all(r.document_id == "doc-repeat" for r in results)


def test_list_by_document_returns_empty_for_unknown(session) -> None:
    """list_by_document() must return [] for an unknown document_id."""
    repo = ExtractionResultRepository(session)
    assert repo.list_by_document("nonexistent") == []


# ──────────────────────────────────────────────────────────────────────────────
# list_by_decision()
# ──────────────────────────────────────────────────────────────────────────────


def test_list_by_decision_filters_correctly(session) -> None:
    """list_by_decision() must only return records matching the decision."""
    repo = ExtractionResultRepository(session)
    repo.save("d1", "t", "auto", "heuristic", "v1", _sample_fields())
    repo.save("d2", "t", "auto", "heuristic", "v1", _sample_fields())
    repo.save("d3", "t", "reject", "heuristic", "v1", _sample_fields())

    auto_records = repo.list_by_decision("auto")
    reject_records = repo.list_by_decision("reject")

    assert len(auto_records) == 2
    assert len(reject_records) == 1
    assert all(r.overall_decision == "auto" for r in auto_records)


# ──────────────────────────────────────────────────────────────────────────────
# count_by_decision()
# ──────────────────────────────────────────────────────────────────────────────


def test_count_by_decision_aggregates_correctly(session) -> None:
    """count_by_decision() must return correct counts per decision label."""
    repo = ExtractionResultRepository(session)
    for decision in ["auto", "auto", "auto", "review", "reject"]:
        repo.save(f"d-{decision}", "t", decision, "heuristic", "v1", _sample_fields())

    counts = repo.count_by_decision()
    assert counts.get("auto", 0) == 3
    assert counts.get("review", 0) == 1
    assert counts.get("reject", 0) == 1


def test_count_by_decision_empty_table(session) -> None:
    """count_by_decision() must return {} on an empty table."""
    repo = ExtractionResultRepository(session)
    assert repo.count_by_decision() == {}


# ──────────────────────────────────────────────────────────────────────────────
# average_confidence_by_field()
# ──────────────────────────────────────────────────────────────────────────────


def test_average_confidence_by_field_is_computed(session) -> None:
    """average_confidence_by_field() must return averaged confidence per label."""
    repo = ExtractionResultRepository(session)
    repo.save("d1", "t", "auto", "heuristic", "v1", [
        FieldRecord("INVOICE_ID", "INV-1", 0.90, "auto", [], "heuristic"),
    ])
    repo.save("d2", "t", "auto", "heuristic", "v1", [
        FieldRecord("INVOICE_ID", "INV-2", 0.80, "auto", [], "heuristic"),
    ])

    avgs = repo.average_confidence_by_field()
    assert "INVOICE_ID" in avgs
    assert avgs["INVOICE_ID"] == pytest.approx(0.85, abs=1e-4)
