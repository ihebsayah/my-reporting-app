"""CLI for ML training utilities."""

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from app.kpi import PipelineKPIService, kpi_report_to_payload
from app.logging_config import configure_logging
from app.ml.ner_extractor import RegexSpacyEnsembleExtractor
from app.ml.ner_trainer import NERMetrics, NERMetricsReport, SpacyNERTrainer
from app.pipeline.batch_processor import PipelineBatchProcessor
from app.pipeline.decision_engine import SequentialExtractionDecisionEngine


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the ML CLI parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description="ML training utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_ner_parser = subparsers.add_parser(
        "train-ner",
        help="Train a spaCy NER model from spaCy JSONL examples.",
    )
    train_ner_parser.add_argument(
        "--input-file",
        default="docs/annotation/spacy_train.jsonl",
        help="spaCy JSONL training data path.",
    )
    train_ner_parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where the trained NER model should be saved.",
    )
    train_ner_parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Number of spaCy training iterations.",
    )

    split_parser = subparsers.add_parser(
        "split-ner-data",
        help="Split spaCy JSONL data into train and validation files.",
    )
    split_parser.add_argument(
        "--input-file",
        default="docs/annotation/spacy_train.jsonl",
        help="spaCy JSONL training data path.",
    )
    split_parser.add_argument(
        "--train-output-file",
        default="docs/annotation/spacy_train_split.jsonl",
        help="Output path for the train split.",
    )
    split_parser.add_argument(
        "--validation-output-file",
        default="docs/annotation/spacy_validation_split.jsonl",
        help="Output path for the validation split.",
    )
    split_parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.2,
        help="Fraction of examples to keep for validation.",
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate-ner-data",
        help="Evaluate predicted spaCy JSONL annotations against gold examples.",
    )
    evaluate_parser.add_argument(
        "--gold-file",
        required=True,
        help="Gold-standard spaCy JSONL path.",
    )
    evaluate_parser.add_argument(
        "--predicted-file",
        required=True,
        help="Predicted spaCy JSONL path.",
    )

    extract_parser = subparsers.add_parser(
        "extract-entities",
        help="Run regex + spaCy ensemble extraction on raw text.",
    )
    extract_parser.add_argument(
        "--text",
        required=True,
        help="Raw document text to analyze.",
    )

    pipeline_parser = subparsers.add_parser(
        "run-pipeline",
        help="Run extraction and threshold-based pipeline decisions on raw text.",
    )
    pipeline_parser.add_argument(
        "--text",
        required=True,
        help="Raw document text to analyze.",
    )

    batch_pipeline_parser = subparsers.add_parser(
        "run-pipeline-batch",
        help="Run the threshold-based pipeline across a directory of source documents.",
    )
    batch_pipeline_parser.add_argument(
        "--input-dir",
        default="docs/source_documents",
        help="Directory containing source documents to process.",
    )

    kpi_parser = subparsers.add_parser(
        "build-kpi-report",
        help="Build reusable KPI metrics from a batch pipeline run.",
    )
    kpi_parser.add_argument(
        "--input-dir",
        default="docs/source_documents",
        help="Directory containing source documents to process.",
    )

    validate_ner_parser = subparsers.add_parser(
        "validate-ner-data",
        help="Validate spaCy JSONL training data without training a model.",
    )
    validate_ner_parser.add_argument(
        "--input-file",
        default="docs/annotation/spacy_train.jsonl",
        help="spaCy JSONL training data path.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the ML CLI.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Process exit code.
    """
    configure_logging()
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    trainer = SpacyNERTrainer()

    if args.command == "train-ner":
        metadata = trainer.train_from_jsonl(
            input_path=Path(args.input_file),
            output_dir=Path(args.output_dir) if args.output_dir else None,
            iterations=args.iterations,
        )
        print(json.dumps(metadata, indent=2))
        return 0

    if args.command == "split-ner-data":
        examples = trainer.load_examples(Path(args.input_file))
        train_examples, validation_examples = trainer.split_examples(
            examples=examples,
            validation_ratio=args.validation_ratio,
        )
        payload = {
            "train_file": trainer.save_examples_jsonl(
                train_examples, Path(args.train_output_file)
            ),
            "validation_file": trainer.save_examples_jsonl(
                validation_examples, Path(args.validation_output_file)
            ),
            "train_examples": len(train_examples),
            "validation_examples": len(validation_examples),
        }
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "evaluate-ner-data":
        gold_examples = trainer.load_examples(Path(args.gold_file))
        predicted_examples = trainer.load_examples(Path(args.predicted_file))
        metrics_report = trainer.evaluate_predictions(gold_examples, predicted_examples)
        print(json.dumps(_metrics_report_to_payload(metrics_report), indent=2))
        return 0

    if args.command == "extract-entities":
        extractor = RegexSpacyEnsembleExtractor()
        result = extractor.extract(args.text)
        payload = {
            "text": result.text,
            "entities": [
                {
                    "start": entity.start,
                    "end": entity.end,
                    "text": entity.text,
                    "label": entity.label,
                    "sources": list(entity.sources),
                    "score": entity.score,
                }
                for entity in result.entities
            ],
        }
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "run-pipeline":
        engine = SequentialExtractionDecisionEngine()
        result = engine.run(args.text)
        payload = {
            "overall_decision": result.overall_decision,
            "fields": [
                {
                    "field_name": field.field_name,
                    "value": field.value,
                    "confidence": field.confidence,
                    "decision": field.decision,
                    "sources": field.sources,
                    "confidence_factors": field.confidence_factors,
                    "start": field.start,
                    "end": field.end,
                }
                for field in result.fields
            ],
        }
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "run-pipeline-batch":
        processor = PipelineBatchProcessor()
        batch_result = processor.run_directory(args.input_dir)
        payload = {
            "documents": [
                {
                    "document_id": document.document_id,
                    "overall_decision": document.result.overall_decision,
                    "field_count": len(document.result.fields),
                }
                for document in batch_result.documents
            ],
            "metrics": {
                "document_count": batch_result.metrics.document_count if batch_result.metrics else 0,
                "overall_decisions": batch_result.metrics.overall_decisions if batch_result.metrics else {},
                "field_decisions": batch_result.metrics.field_decisions if batch_result.metrics else {},
            },
        }
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "build-kpi-report":
        processor = PipelineBatchProcessor()
        batch_result = processor.run_directory(args.input_dir)
        report = PipelineKPIService().build_report(batch_result)
        print(json.dumps(kpi_report_to_payload(report), indent=2))
        return 0

    if args.command == "validate-ner-data":
        examples = trainer.load_examples(Path(args.input_file))
        errors = trainer.validate_examples(examples)
        payload = {"examples": len(examples), "errors": errors}
        print(json.dumps(payload, indent=2))
        return 0 if not errors else 1

    parser.error(f"Unknown command: {args.command}")
    return 2


def _metrics_to_payload(metrics: NERMetrics) -> dict:
    """Serialize NER metrics for CLI output."""
    return {
        "true_positives": metrics.true_positives,
        "false_positives": metrics.false_positives,
        "false_negatives": metrics.false_negatives,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1_score": metrics.f1_score,
    }


def _metrics_report_to_payload(report: NERMetricsReport) -> dict:
    """Serialize aggregate and per-label NER metrics for CLI output."""
    overall = _metrics_to_payload(report.overall)
    return {
        **overall,
        "overall": overall,
        "per_label": {
            label: _metrics_to_payload(metrics)
            for label, metrics in report.per_label.items()
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
