"""CLI for ML training utilities."""

import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence

from app.kpi import PipelineKPIService, kpi_report_to_payload
from app.logging_config import configure_logging
from app.ml.feature_builder import EntityFeatureBuilder
from app.ml.ner_extractor import RegexSpacyEnsembleExtractor
from app.ml.ner_trainer import NERMetrics, NERMetricsReport, SpacyNERTrainer
from app.ml.rf_confidence_model import RFConfidenceModel, TrainingRecord
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

    # ── RF confidence model ────────────────────────────────────────────────
    train_rf_parser = subparsers.add_parser(
        "train-rf",
        help=(
            "Train the Random Forest confidence model from a labeled JSONL file. "
            "Each line must have: document_id, text, entities, is_correct (bool)."
        ),
    )
    train_rf_parser.add_argument(
        "--input-file",
        default="docs/annotation/rf_training_records.jsonl",
        help="Labeled JSONL file with TrainingRecord entries.",
    )
    train_rf_parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where the versioned .joblib model is saved.",
    )
    train_rf_parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="Number of trees in the Random Forest (default: 200).",
    )
    train_rf_parser.add_argument(
        "--no-bert",
        action="store_true",
        default=False,
        help="Disable BERT embeddings and use hand-crafted features only.",
    )

    score_rf_parser = subparsers.add_parser(
        "score-rf",
        help="Extract entities from raw text and score them with the RF confidence model.",
    )
    score_rf_parser.add_argument(
        "--text",
        required=True,
        help="Raw document text to analyse.",
    )
    score_rf_parser.add_argument(
        "--model-path",
        default=None,
        help="Path to a trained RF .joblib file. Uses settings default when omitted.",
    )
    score_rf_parser.add_argument(
        "--no-bert",
        action="store_true",
        default=False,
        help="Disable BERT embeddings (must match the setting used during training).",
    )

    # ── Monthly retraining ─────────────────────────────────────────────────
    retrain_rf_parser = subparsers.add_parser(
        "retrain-rf",
        help=(
            "Run the monthly RF retraining pipeline: merges base annotations with "
            "human feedback corrections and saves a new versioned model."
        ),
    )
    retrain_rf_parser.add_argument(
        "--input-file",
        default="docs/annotation/rf_training_records.jsonl",
        help="Base labeled JSONL training file.",
    )
    retrain_rf_parser.add_argument(
        "--feedback-file",
        default=None,
        help="Path to feedback.jsonl. Defaults to artifacts/feedback/feedback.jsonl.",
    )
    retrain_rf_parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save the new versioned RF model. Uses settings default.",
    )
    retrain_rf_parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="Number of trees in the retrained RF (default: 200).",
    )
    retrain_rf_parser.add_argument(
        "--no-bert",
        action="store_true",
        default=False,
        help="Disable BERT embeddings during retraining.",
    )

    # ── Drift monitoring ───────────────────────────────────────────────────
    drift_parser = subparsers.add_parser(
        "check-drift",
        help=(
            "Check for confidence and auto-rate drift by querying the DB "
            "and comparing baseline vs recent sliding windows."
        ),
    )
    drift_parser.add_argument(
        "--baseline-window",
        type=int,
        default=50,
        help="Number of records in the baseline window (default: 50).",
    )
    drift_parser.add_argument(
        "--recent-window",
        type=int,
        default=20,
        help="Number of records in the recent window (default: 20).",
    )
    drift_parser.add_argument(
        "--auto-threshold",
        type=float,
        default=0.10,
        help="Auto-rate drop threshold to trigger drift (default: 0.10).",
    )
    drift_parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.05,
        help="Confidence drop threshold to trigger drift (default: 0.05).",
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

    if args.command == "train-rf":
        records = _load_rf_training_records(Path(args.input_file))
        rf_model = RFConfidenceModel(use_bert=not args.no_bert)
        result = rf_model.train(
            records,
            output_dir=Path(args.output_dir) if args.output_dir else None,
            n_estimators=args.n_estimators,
        )
        print(
            json.dumps(
                {
                    "model_version": result.model_version,
                    "model_path": result.model_path,
                    "train_samples": result.train_samples,
                    "accuracy": round(result.accuracy, 4),
                    "feature_count": result.feature_count,
                    "labels": result.labels,
                    "trained_at": result.trained_at,
                },
                indent=2,
            )
        )
        return 0

    if args.command == "score-rf":
        extractor = RegexSpacyEnsembleExtractor()
        extraction = extractor.extract(args.text)
        if not extraction.entities:
            print(json.dumps({"message": "No entities extracted.", "entities": []}, indent=2))
            return 0

        rf_model = RFConfidenceModel(use_bert=not args.no_bert)
        model_path_str = args.model_path or rf_model.settings.rf_model_path
        model_path = Path(model_path_str)

        # Attempt to auto-discover the latest .joblib in a directory.
        if model_path.is_dir():
            candidates = sorted(model_path.glob("rf_confidence_v*.joblib"), reverse=True)
            if not candidates:
                print(
                    json.dumps(
                        {"error": f"No RF model found in {model_path}. Run train-rf first."},
                        indent=2,
                    )
                )
                return 1
            model_path = candidates[0]

        rf_model.load(model_path)
        predictions = rf_model.predict_batch(extraction.entities, args.text)
        payload = {
            "text": args.text,
            "model_version": predictions[0].model_version if predictions else None,
            "predictions": [
                {
                    "label": p.entity_label,
                    "value": p.entity_text,
                    "confidence": round(p.confidence, 4),
                    "decision": p.decision,
                }
                for p in predictions
            ],
        }
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "retrain-rf":
        from app.retraining.pipeline import RFRetrainingPipeline

        retrain_pipeline = RFRetrainingPipeline(use_bert=not args.no_bert)
        result = retrain_pipeline.run(
            base_jsonl_path=Path(args.input_file),
            output_dir=Path(args.output_dir) if args.output_dir else None,
            feedback_path=Path(args.feedback_file) if args.feedback_file else None,
            n_estimators=args.n_estimators,
        )
        print(
            json.dumps(
                {
                    "model_version": result.training_result.model_version,
                    "model_path": result.model_path,
                    "accuracy": result.training_result.accuracy,
                    "base_records": result.base_records,
                    "feedback_records": result.feedback_records,
                    "total_records": result.total_records,
                    "feedback_file_exists": result.feedback_file_exists,
                },
                indent=2,
            )
        )
        return 0

    if args.command == "check-drift":
        from app.config import get_settings
        from app.database import ExtractionResultRepository, get_session_factory, init_database
        from app.monitoring.drift import ConfidenceDriftDetector

        drift_settings = get_settings()
        init_database(drift_settings)
        session_factory = get_session_factory(drift_settings)
        needed = args.baseline_window + args.recent_window

        with session_factory() as session:
            repo = ExtractionResultRepository(session)
            records = (
                repo.list_by_decision("auto", limit=needed)
                + repo.list_by_decision("review", limit=needed)
                + repo.list_by_decision("reject", limit=needed)
            )

        detector = ConfidenceDriftDetector(
            auto_rate_drop_threshold=args.auto_threshold,
            confidence_drop_threshold=args.confidence_threshold,
            baseline_window_size=args.baseline_window,
            recent_window_size=args.recent_window,
        )
        samples = detector.samples_from_records(records)
        report = detector.detect(samples)
        print(
            json.dumps(
                {
                    "drift_detected": report.drift_detected,
                    "triggered_signals": report.triggered_signals,
                    "auto_rate_drop": report.auto_rate_drop,
                    "confidence_drop": report.confidence_drop,
                    "baseline": {
                        "window_size": report.baseline.window_size,
                        "auto_rate": report.baseline.auto_rate,
                        "mean_confidence": report.baseline.mean_confidence,
                    },
                    "recent": {
                        "window_size": report.recent.window_size,
                        "auto_rate": report.recent.auto_rate,
                        "mean_confidence": report.recent.mean_confidence,
                    },
                    "checked_at": report.checked_at,
                },
                indent=2,
            )
        )
        return 0

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


def _load_rf_training_records(input_path: Path) -> List[TrainingRecord]:
    """Load RF confidence model training records from a labeled JSONL file.

    Each JSONL line must be a JSON object with:
      * ``text`` (str) – the document text.
      * ``entities`` (list of [start, end, label]) – NER spans.
      * ``is_correct`` (bool) – human-verified correctness label.
      * ``document_id`` (str, optional) – identifier used as entity source.

    Args:
        input_path: Path to the labeled JSONL file.

    Returns:
        List of ``TrainingRecord`` objects ready for ``RFConfidenceModel.train()``.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If any line is malformed.
    """
    from app.ml.ner_extractor import ExtractedEntity

    if not input_path.exists():
        raise FileNotFoundError(
            f"RF training records file not found: {input_path}\n"
            "Create it by exporting annotations with `annotation export-annotations` "
            "and labeling each entity with `is_correct`."
        )

    records: List[TrainingRecord] = []
    for line_number, raw_line in enumerate(
        input_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Malformed JSON on line {line_number} of {input_path}: {exc}"
            ) from exc

        text = str(payload.get("text", ""))
        is_correct = bool(payload.get("is_correct", False))
        doc_id = str(payload.get("document_id", f"line-{line_number}"))

        for entity_payload in payload.get("entities", []):
            start, end, label = int(entity_payload[0]), int(entity_payload[1]), str(entity_payload[2])
            entity = ExtractedEntity(
                start=start,
                end=end,
                text=text[start:end],
                label=label,
                sources=("annotation",),
                score=1.0 if is_correct else 0.0,
            )
            records.append(
                TrainingRecord(
                    entity=entity,
                    document_text=text,
                    is_correct=is_correct,
                )
            )

    if not records:
        raise ValueError(
            f"No training records parsed from {input_path}. "
            "Ensure the file has lines with 'text', 'entities', and 'is_correct' keys."
        )
    return records


if __name__ == "__main__":
    raise SystemExit(main())

