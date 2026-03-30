"""Helpers for building Label Studio task payloads from normalized documents."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from app.file_processing.document_extractor import DocumentExtractor
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceDocument:
    """Represents a document prepared for annotation."""

    document_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def load_documents_from_directory(input_dir: Path) -> List[SourceDocument]:
    """Load annotation-ready source documents from a directory.

    Args:
        input_dir: Directory containing text or JSON source files.

    Returns:
        Loaded source documents sorted by filename.

    Raises:
        FileNotFoundError: If the input directory does not exist.
        ValueError: If a source file is malformed.
    """
    extractor = DocumentExtractor()
    extracted_documents = extractor.extract_directory(input_dir)

    documents: List[SourceDocument] = [
        SourceDocument(
            document_id=document.document_id,
            text=document.text,
            metadata={
                **document.metadata,
                "source_path": document.source_path,
                "file_type": document.file_type,
            },
        )
        for document in extracted_documents
    ]

    logger.info("Loaded %d source documents from %s.", len(documents), input_dir)
    return documents


def build_label_studio_tasks(documents: List[SourceDocument]) -> List[Dict[str, Any]]:
    """Convert source documents into Label Studio import payloads.

    Args:
        documents: Source documents ready for annotation.

    Returns:
        Label Studio task payloads using `data.text` and document metadata.
    """
    tasks: List[Dict[str, Any]] = []
    for document in documents:
        task: Dict[str, Any] = {
            "data": {
                "text": document.text,
                "document_id": document.document_id,
            },
            "meta": document.metadata,
        }
        tasks.append(task)
    logger.info("Built %d Label Studio task payloads.", len(tasks))
    return tasks
