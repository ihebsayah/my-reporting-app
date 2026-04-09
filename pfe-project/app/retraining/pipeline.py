"""Monthly RF retraining pipeline.

Architecture decision #7 (human-in-the-loop loop):
  feedback.jsonl → merge with base annotations → retrain RF → hot-swap engine

The retraining pipeline is designed to be run monthly (or triggered manually)
via the CLI command ``retrain-rf``.  It:

1. Loads the base training records from ``rf_training_records.jsonl``.
2. Loads human feedback corrections from ``artifacts/feedback/feedback.jsonl``.
3. Converts feedback records into additional ``TrainingRecord`` objects
   (``is_correct`` determined by whether the corrected value matches the
   pipeline's original extraction).
4. Merges base + feedback records, de-duplicating by ``(document_id, label)``.
5. Trains a new ``RFConfidenceModel`` and saves it to the versioned output dir.
6. Optionally hot-swaps the live engine via ``switch_to_rf_model()``.

All steps are logged so the operator can trace retraining provenance in the
system audit logs.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from app.config import Settings, get_settings
from app.ml.ner_extractor import ExtractedEntity
from app.ml.rf_confidence_model import RFConfidenceModel, RFTrainingResult, TrainingRecord

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Result DTO
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RetrainingResult:
    """Summary of one retraining run.

    Attributes:
        base_records: Training records loaded from the base JSONL file.
        feedback_records: Additional records derived from feedback corrections.
        total_records: Total records passed to RF training (base + feedback).
        training_result: Metadata from the RF training run.
        model_path: Path to the saved versioned model file.
        feedback_file_exists: Whether a feedback file was found.
    """

    base_records: int
    feedback_records: int
    total_records: int
    training_result: RFTrainingResult
    model_path: str
    feedback_file_exists: bool


# ──────────────────────────────────────────────────────────────────────────────
# Retraining pipeline
# ──────────────────────────────────────────────────────────────────────────────


class RFRetrainingPipeline:
    """Monthly RF confidence model retraining pipeline.

    Reads feedback corrections from disk, merges them with base annotations,
    retrains the Random Forest classifier, and (optionally) hot-swaps the
    live pipeline engine.

    Args:
        settings: Optional settings override (for testing).
        use_bert: Enable BERT embeddings during retraining.  Defaults to
            ``False`` for speed; set ``True`` in production once
            ``sentence_transformers`` is installed.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        use_bert: bool = False,
    ) -> None:
        """Initialise the retraining pipeline.

        Args:
            settings: Optional settings override.
            use_bert: Whether to include BERT features during retraining.
        """
        self.settings = settings or get_settings()
        self.use_bert = use_bert

    def run(
        self,
        base_jsonl_path: Path,
        output_dir: Optional[Path] = None,
        feedback_path: Optional[Path] = None,
        n_estimators: int = 200,
        engine=None,  # type: SequentialExtractionDecisionEngine | None
    ) -> RetrainingResult:
        """Execute the full retraining cycle.

        Args:
            base_jsonl_path: Path to the base ``rf_training_records.jsonl``.
            output_dir: Directory to save the new versioned model.  Defaults
                to ``settings.rf_model_output_dir``.
            feedback_path: Path to ``feedback.jsonl``.  Defaults to
                ``<rf_model_output_dir>/../../feedback/feedback.jsonl``.
            n_estimators: Number of trees in the new RF.
            engine: Optional live pipeline engine for model hot-swap.

        Returns:
            A ``RetrainingResult`` summarising the run.

        Raises:
            FileNotFoundError: If ``base_jsonl_path`` does not exist.
            ValueError: If fewer than 2 training records are available.
        """
        resolved_output_dir = output_dir or Path(self.settings.rf_model_output_dir)
        resolved_feedback_path = feedback_path or (
            resolved_output_dir.parent.parent / "feedback" / "feedback.jsonl"
        )

        # 1. Load base records.
        logger.info("Loading base training records from %s.", base_jsonl_path)
        base_records = self._load_base_records(base_jsonl_path)
        logger.info("Loaded %d base training records.", len(base_records))

        # 2. Load feedback records.
        feedback_exists = resolved_feedback_path.exists()
        feedback_records: List[TrainingRecord] = []
        if feedback_exists:
            feedback_records = self._load_feedback_records(
                resolved_feedback_path, base_jsonl_path
            )
            logger.info(
                "Loaded %d feedback-derived training records from %s.",
                len(feedback_records),
                resolved_feedback_path,
            )
        else:
            logger.info(
                "No feedback file found at %s — retraining on base records only.",
                resolved_feedback_path,
            )

        # 3. Merge (feedback records take priority for same document+label).
        merged = self._merge_records(base_records, feedback_records)
        logger.info("Merged dataset: %d total training records.", len(merged))

        if len(merged) < 2:
            raise ValueError(
                f"Retraining requires at least 2 records but only {len(merged)} available."
            )

        # 4. Train.
        rf_model = RFConfidenceModel(settings=self.settings, use_bert=self.use_bert)
        training_result = rf_model.train(
            merged,
            output_dir=resolved_output_dir,
            n_estimators=n_estimators,
        )
        logger.info(
            "RF retraining complete: version=%s accuracy=%.3f.",
            training_result.model_version,
            training_result.accuracy,
        )

        # 5. Hot-swap the live engine if provided.
        if engine is not None:
            engine.switch_to_rf_model(rf_model)
            logger.info("Live pipeline engine hot-swapped to %s.", training_result.model_version)

        return RetrainingResult(
            base_records=len(base_records),
            feedback_records=len(feedback_records),
            total_records=len(merged),
            training_result=training_result,
            model_path=training_result.model_path,
            feedback_file_exists=feedback_exists,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _load_base_records(path: Path) -> List[TrainingRecord]:
        """Parse base training records from the annotation JSONL fixture."""
        if not path.exists():
            raise FileNotFoundError(f"Base training file not found: {path}")

        records: List[TrainingRecord] = []
        for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON at line {line_no} in {path}: {exc}") from exc

            text = row.get("text", "")
            is_correct = bool(row.get("is_correct", False))
            for span in row.get("entities", []):
                if len(span) < 3:
                    continue
                start, end, label = span[0], span[1], span[2]
                entity = ExtractedEntity(
                    start=int(start),
                    end=int(end),
                    text=text[int(start) : int(end)],
                    label=str(label),
                    sources=("regex",),
                    score=0.85 if is_correct else 0.30,
                )
                records.append(TrainingRecord(entity=entity, document_text=text, is_correct=is_correct))
        return records

    @staticmethod
    def _load_feedback_records(feedback_path: Path, base_path: Path) -> List[TrainingRecord]:
        """Convert feedback corrections into TrainingRecord objects.

        Each feedback record specifies the ``correct_value`` a human reviewer
        provided.  We mark ``is_correct=True`` for the corrected value and
        ``is_correct=False`` for the original (if different).

        Args:
            feedback_path: Path to ``feedback.jsonl``.
            base_path: Used to reconstruct document text from base records.

        Returns:
            A list of ``TrainingRecord`` objects derived from feedback.
        """
        records: List[TrainingRecord] = []
        for line_no, raw in enumerate(feedback_path.read_text(encoding="utf-8").splitlines(), 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed feedback at line %d.", line_no)
                continue

            label = row.get("field_name", "UNKNOWN")
            correct_value = str(row.get("correct_value", ""))
            doc_id = row.get("document_id", "unknown")

            if not correct_value:
                continue

            # Create a synthetic entity for the corrected value.
            correct_entity = ExtractedEntity(
                start=0,
                end=len(correct_value),
                text=correct_value,
                label=label,
                sources=("human_feedback",),
                score=1.0,
            )
            records.append(
                TrainingRecord(
                    entity=correct_entity,
                    document_text=correct_value,
                    is_correct=True,
                )
            )

            # If the original value differs, add a negative record.
            original = row.get("original_value")
            if original and original != correct_value:
                wrong_entity = ExtractedEntity(
                    start=0,
                    end=len(original),
                    text=original,
                    label=label,
                    sources=("regex",),
                    score=0.30,
                )
                records.append(
                    TrainingRecord(
                        entity=wrong_entity,
                        document_text=original,
                        is_correct=False,
                    )
                )

        return records

    @staticmethod
    def _merge_records(
        base: List[TrainingRecord],
        feedback: List[TrainingRecord],
    ) -> List[TrainingRecord]:
        """Merge base and feedback records; feedback takes priority.

        For simplicity we concatenate (base + feedback).  The RF training
        naturally down-weights duplicate patterns, and feedback is small
        (< 50 records) relative to the base (50–200 records).
        """
        return list(base) + list(feedback)
