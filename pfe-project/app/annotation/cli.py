"""Command-line helpers for Month 1 annotation bootstrap tasks."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from app.annotation.bootstrap import build_annotation_bootstrap_artifacts
from app.annotation.client import LabelStudioClient
from app.annotation.exporter import (
    load_training_export,
    parse_label_studio_export,
    save_spacy_jsonl,
    save_training_export,
)
from app.annotation.quality import build_presence_comparisons, summarize_field_agreement
from app.annotation.task_builder import build_label_studio_tasks, load_documents_from_directory
from app.config import get_settings
from app.logging_config import configure_logging

logger = logging.getLogger(__name__)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for annotation bootstrap operations.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Annotation bootstrap utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser(
        "generate-assets",
        help="Generate schema, Label Studio config, and annotation guidelines.",
    )
    generate_parser.add_argument(
        "--output-dir",
        default="docs/annotation",
        help="Directory where generated annotation assets will be written.",
    )

    create_parser = subparsers.add_parser(
        "create-project",
        help="Generate assets and create the Label Studio project remotely.",
    )
    create_parser.add_argument(
        "--output-dir",
        default="docs/annotation",
        help="Directory where generated annotation assets will be written.",
    )

    import_parser = subparsers.add_parser(
        "import-tasks",
        help="Load local source documents and import them into Label Studio.",
    )
    import_parser.add_argument("--project-id", type=int, required=True)
    import_parser.add_argument(
        "--input-dir",
        default="docs/source_documents",
        help="Directory containing .txt or .json source documents.",
    )

    export_parser = subparsers.add_parser(
        "export-annotations",
        help="Export annotations from Label Studio and save training-ready JSON.",
    )
    export_parser.add_argument("--project-id", type=int, required=True)
    export_parser.add_argument(
        "--output-file",
        default="docs/annotation/training_export.json",
        help="Path where training-ready JSON should be written.",
    )
    export_parser.add_argument(
        "--raw-output-file",
        default="docs/annotation/label_studio_export.json",
        help="Path where the raw Label Studio export should be written.",
    )

    spacy_parser = subparsers.add_parser(
        "export-spacy",
        help="Convert normalized training JSON into spaCy JSONL training data.",
    )
    spacy_parser.add_argument(
        "--input-file",
        default="docs/annotation/training_export.json",
        help="Path to normalized training export JSON.",
    )
    spacy_parser.add_argument(
        "--output-file",
        default="docs/annotation/spacy_train.jsonl",
        help="Path where spaCy JSONL should be written.",
    )

    agreement_parser = subparsers.add_parser(
        "agreement-report",
        help="Compute field presence agreement from two JSON annotation summaries.",
    )
    agreement_parser.add_argument(
        "--annotator-a", required=True, help="Path to annotator A JSON summary."
    )
    agreement_parser.add_argument(
        "--annotator-b", required=True, help="Path to annotator B JSON summary."
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the annotation bootstrap CLI.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Process exit code.
    """
    configure_logging()
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    if args.command == "generate-assets":
        return _handle_generate_assets(Path(args.output_dir))
    if args.command == "create-project":
        return _handle_create_project(Path(args.output_dir))
    if args.command == "import-tasks":
        return _handle_import_tasks(args.project_id, Path(args.input_dir))
    if args.command == "export-annotations":
        return _handle_export_annotations(
            args.project_id,
            Path(args.output_file),
            Path(args.raw_output_file),
        )
    if args.command == "export-spacy":
        return _handle_export_spacy(Path(args.input_file), Path(args.output_file))
    if args.command == "agreement-report":
        return _handle_agreement_report(Path(args.annotator_a), Path(args.annotator_b))

    parser.error(f"Unknown command: {args.command}")
    return 2


def _handle_generate_assets(output_dir: Path) -> int:
    """Generate the local annotation bootstrap assets."""
    artifact_paths = build_annotation_bootstrap_artifacts(output_dir=output_dir)
    print(json.dumps(artifact_paths, indent=2))
    return 0


def _handle_create_project(output_dir: Path) -> int:
    """Generate assets and create a remote Label Studio project."""
    settings = get_settings()
    artifact_paths = build_annotation_bootstrap_artifacts(output_dir=output_dir)
    label_config = Path(artifact_paths["label_config"]).read_text(encoding="utf-8")
    client = LabelStudioClient(settings=settings)
    response = client.create_project(
        title=settings.label_studio_project_title,
        label_config=label_config,
    )
    print(json.dumps({"artifacts": artifact_paths, "project": response}, indent=2))
    return 0


def _handle_import_tasks(project_id: int, input_dir: Path) -> int:
    """Load local documents and import them into Label Studio."""
    documents = load_documents_from_directory(input_dir)
    tasks = build_label_studio_tasks(documents)
    client = LabelStudioClient(settings=get_settings())
    response = client.import_tasks(project_id=project_id, tasks=tasks)
    print(
        json.dumps(
            {
                "project_id": project_id,
                "input_documents": len(documents),
                "import_response": response,
            },
            indent=2,
        )
    )
    return 0


def _handle_export_annotations(
    project_id: int, output_file: Path, raw_output_file: Path
) -> int:
    """Export Label Studio annotations and save training-ready records."""
    client = LabelStudioClient(settings=get_settings())
    raw_payload = client.export_annotations(project_id=project_id)
    raw_output_file.parent.mkdir(parents=True, exist_ok=True)
    raw_output_file.write_text(json.dumps(raw_payload, indent=2), encoding="utf-8")
    documents = parse_label_studio_export(raw_payload)
    saved_paths = save_training_export(
        documents=documents,
        output_path=output_file,
    )
    print(
        json.dumps(
            {
                "project_id": project_id,
                "documents": len(documents),
                "saved_paths": {
                    **saved_paths,
                    "raw": str(raw_output_file),
                },
            },
            indent=2,
        )
    )
    return 0


def _handle_export_spacy(input_file: Path, output_file: Path) -> int:
    """Convert normalized annotation exports into spaCy JSONL."""
    documents = load_training_export(input_file)
    saved_paths = save_spacy_jsonl(documents=documents, output_path=output_file)
    print(
        json.dumps(
            {
                "documents": len(documents),
                "saved_paths": saved_paths,
            },
            indent=2,
        )
    )
    return 0


def _handle_agreement_report(annotator_a_path: Path, annotator_b_path: Path) -> int:
    """Compute and print a field-level agreement report."""
    annotator_a = _load_annotation_summary(annotator_a_path)
    annotator_b = _load_annotation_summary(annotator_b_path)
    comparisons = build_presence_comparisons(annotator_a, annotator_b)
    results = summarize_field_agreement(comparisons)
    payload: Dict[str, Any] = {
        field_name: {
            "observed_agreement": result.observed_agreement,
            "expected_agreement": result.expected_agreement,
            "cohen_kappa": result.cohen_kappa,
            "comparisons": result.comparisons,
        }
        for field_name, result in sorted(results.items())
    }
    print(json.dumps(payload, indent=2))
    return 0


def _load_annotation_summary(path: Path) -> Dict[str, Sequence[str]]:
    """Load a document-to-field summary JSON file.

    Args:
        path: JSON file path.

    Returns:
        Mapping of document IDs to annotated field lists.

    Raises:
        ValueError: If the JSON structure is invalid.
    """
    logger.info("Loading annotation summary from %s.", path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        logger.error("Annotation summary at %s is not a JSON object.", path)
        raise ValueError(
            "Annotation summary must be a JSON object keyed by document ID."
        )
    for document_id, fields in payload.items():
        if not isinstance(document_id, str) or not isinstance(fields, list):
            logger.error(
                "Invalid annotation summary entry for document %s.", document_id
            )
            raise ValueError(
                "Each annotation summary value must be a list of field names."
            )
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
