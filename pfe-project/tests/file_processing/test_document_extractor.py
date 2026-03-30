"""Tests for raw document extraction."""

import json
from pathlib import Path

import pytest

from app.file_processing.document_extractor import (
    DocumentExtractionError,
    DocumentExtractor,
)


def test_extract_directory_supports_txt_json_and_csv(tmp_path: Path) -> None:
    """Ensure the extractor normalizes supported local source files."""
    (tmp_path / "sample.txt").write_text("hello world", encoding="utf-8")
    (tmp_path / "sample.json").write_text(
        json.dumps(
            {
                "document_id": "json-doc",
                "text": "json payload text",
                "metadata": {"origin": "fixture"},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "sample.csv").write_text(
        "field,value\ninvoice_id,INV-001\n",
        encoding="utf-8",
    )

    extractor = DocumentExtractor()
    documents = extractor.extract_directory(tmp_path)

    assert len(documents) == 3
    assert documents[0].file_type == "csv"
    assert documents[1].document_id == "json-doc"
    assert documents[2].text == "hello world"


def test_extract_csv_flattens_rows_into_text(tmp_path: Path) -> None:
    """Ensure CSV rows become labeled text lines."""
    path = tmp_path / "invoice.csv"
    path.write_text("field,value\ninvoice_id,INV-001\n", encoding="utf-8")

    document = DocumentExtractor().extract_file(path)

    assert document.file_type == "csv"
    assert "row_1: field | value" in document.text
    assert "row_2: invoice_id | INV-001" in document.text


def test_extract_pdf_without_dependency_raises_clean_error(tmp_path: Path) -> None:
    """Ensure optional PDF support fails with a helpful wrapped error."""
    path = tmp_path / "invoice.pdf"
    path.write_bytes(b"%PDF-1.4\n")

    with pytest.raises(DocumentExtractionError) as exc_info:
        DocumentExtractor().extract_file(path)

    assert "pypdf is required" in str(exc_info.value)
