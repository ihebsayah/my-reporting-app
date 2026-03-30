"""Helpers for converting annotations into training-friendly records."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpanAnnotation:
    """Represents one annotated span in a document."""

    start: int
    end: int
    text: str
    label: str


@dataclass(frozen=True)
class AnnotatedDocument:
    """Represents a document and its span annotations."""

    document_id: str
    text: str
    annotations: List[SpanAnnotation]


def export_for_training(documents: List[AnnotatedDocument]) -> List[Dict[str, Any]]:
    """Convert annotated documents into JSON-friendly training records.

    Args:
        documents: Documents to export.

    Returns:
        A list of dictionaries ready for later ML preprocessing.
    """
    logger.info("Exporting %d annotated documents for downstream training.", len(documents))
    return [
        {
            "document_id": document.document_id,
            "text": document.text,
            "annotations": [
                {
                    "start": annotation.start,
                    "end": annotation.end,
                    "text": annotation.text,
                    "label": annotation.label,
                }
                for annotation in document.annotations
            ],
        }
        for document in documents
    ]


def parse_label_studio_export(payload: List[Dict[str, Any]]) -> List[AnnotatedDocument]:
    """Convert Label Studio JSON export payload into normalized annotations.

    Args:
        payload: Raw Label Studio task export payload.

    Returns:
        Normalized annotated documents for downstream training.
    """
    documents: List[AnnotatedDocument] = []
    for task in payload:
        data = task.get("data", {})
        text = str(data.get("text", ""))
        document_id = str(data.get("document_id") or task.get("id") or "unknown-document")
        annotations = _extract_span_annotations(task.get("annotations", []), text)
        documents.append(
            AnnotatedDocument(
                document_id=document_id,
                text=text,
                annotations=annotations,
            )
        )
    logger.info("Parsed %d annotated documents from Label Studio export.", len(documents))
    return documents


def save_training_export(
    documents: List[AnnotatedDocument],
    output_path: Path,
) -> Dict[str, str]:
    """Persist normalized training records to disk.

    Args:
        documents: Normalized documents to export.
        output_path: JSON path for training-ready records.

    Returns:
        Paths of saved export artifacts.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    training_payload = export_for_training(documents)
    output_path.write_text(json.dumps(training_payload, indent=2), encoding="utf-8")

    logger.info("Saved training export to %s.", output_path)
    return {"training": str(output_path)}


def convert_to_spacy_examples(
    documents: List[AnnotatedDocument],
) -> List[Dict[str, Any]]:
    """Convert annotated documents into spaCy NER training examples.

    Args:
        documents: Normalized annotated documents.

    Returns:
        A list of JSON-serializable spaCy training examples.
    """
    examples: List[Dict[str, Any]] = []
    for document in documents:
        entities = sorted(
            [
                [annotation.start, annotation.end, annotation.label]
                for annotation in document.annotations
            ],
            key=lambda item: (item[0], item[1], item[2]),
        )
        examples.append(
            {
                "document_id": document.document_id,
                "text": document.text,
                "entities": entities,
            }
        )
    logger.info("Converted %d documents into spaCy training examples.", len(examples))
    return examples


def save_spacy_jsonl(
    documents: List[AnnotatedDocument],
    output_path: Path,
) -> Dict[str, str]:
    """Save spaCy-style training examples as JSONL.

    Args:
        documents: Normalized annotated documents.
        output_path: Destination JSONL path.

    Returns:
        Paths of saved export artifacts.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    examples = convert_to_spacy_examples(documents)
    lines = [json.dumps(example, ensure_ascii=True) for example in examples]
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    logger.info("Saved spaCy JSONL export to %s.", output_path)
    return {"spacy_jsonl": str(output_path)}


def load_training_export(input_path: Path) -> List[AnnotatedDocument]:
    """Load normalized training records from disk.

    Args:
        input_path: JSON path created by `save_training_export`.

    Returns:
        Parsed annotated documents.
    """
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    documents: List[AnnotatedDocument] = []
    for item in payload:
        annotations = [
            SpanAnnotation(
                start=int(annotation["start"]),
                end=int(annotation["end"]),
                text=str(annotation["text"]),
                label=str(annotation["label"]),
            )
            for annotation in item.get("annotations", [])
        ]
        documents.append(
            AnnotatedDocument(
                document_id=str(item["document_id"]),
                text=str(item["text"]),
                annotations=annotations,
            )
        )
    logger.info("Loaded %d normalized training documents from %s.", len(documents), input_path)
    return documents


def _extract_span_annotations(
    raw_annotations: List[Dict[str, Any]], text: str
) -> List[SpanAnnotation]:
    """Extract span annotations from Label Studio results."""
    spans: List[SpanAnnotation] = []
    for annotation in raw_annotations:
        for result in annotation.get("result", []):
            value = result.get("value", {})
            labels = value.get("labels", [])
            start = value.get("start")
            end = value.get("end")
            if (
                result.get("type") != "labels"
                or not labels
                or start is None
                or end is None
            ):
                continue
            span_text = str(value.get("text", text[int(start) : int(end)]))
            spans.append(
                SpanAnnotation(
                    start=int(start),
                    end=int(end),
                    text=span_text,
                    label=str(labels[0]),
                )
            )
    return spans
