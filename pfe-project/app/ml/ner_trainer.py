"""spaCy-oriented NER training helpers."""

import importlib
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpacyTrainingExample:
    """Represents one spaCy-style NER training example."""

    document_id: str
    text: str
    entities: List[Tuple[int, int, str]]


class NERTrainingError(RuntimeError):
    """Raised when NER training setup or execution fails."""


@dataclass(frozen=True)
class NERMetrics:
    """Entity-level NER precision, recall, and F1 metrics."""

    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float


@dataclass(frozen=True)
class NERMetricsReport:
    """Aggregate and per-label NER metrics."""

    overall: NERMetrics
    per_label: Dict[str, NERMetrics]


class SpacyNERTrainer:
    """Load, validate, and optionally train a spaCy NER model."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """Initialize the trainer.

        Args:
            settings: Optional application settings override.
        """
        self.settings = settings or get_settings()

    def load_examples(self, input_path: Path) -> List[SpacyTrainingExample]:
        """Load spaCy JSONL examples from disk.

        Args:
            input_path: JSONL file path.

        Returns:
            Parsed training examples.
        """
        examples: List[SpacyTrainingExample] = []
        for line_number, line in enumerate(
            input_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            payload = json.loads(line)
            entities = [
                (int(item[0]), int(item[1]), str(item[2]))
                for item in payload.get("entities", [])
            ]
            examples.append(
                SpacyTrainingExample(
                    document_id=str(payload.get("document_id", f"line-{line_number}")),
                    text=str(payload["text"]),
                    entities=entities,
                )
            )
        logger.info("Loaded %d spaCy training examples from %s.", len(examples), input_path)
        return examples

    def split_examples(
        self,
        examples: Sequence[SpacyTrainingExample],
        validation_ratio: float = 0.2,
        random_seed: int = 42,
    ) -> Tuple[List[SpacyTrainingExample], List[SpacyTrainingExample]]:
        """Split examples into train and validation sets.

        Args:
            examples: Full set of labeled examples.
            validation_ratio: Fraction to allocate to validation.
            random_seed: Seed for deterministic shuffling.

        Returns:
            Train and validation example lists.

        Raises:
            ValueError: If the split ratio is invalid or not enough examples exist.
        """
        if not 0.0 < validation_ratio < 1.0:
            raise ValueError("validation_ratio must be between 0 and 1.")
        if len(examples) < 2:
            raise ValueError("At least two examples are required to create a split.")

        shuffled_examples = list(examples)
        random.Random(random_seed).shuffle(shuffled_examples)
        validation_count = max(1, int(round(len(shuffled_examples) * validation_ratio)))
        validation_examples = shuffled_examples[:validation_count]
        train_examples = shuffled_examples[validation_count:]
        if not train_examples or not validation_examples:
            raise ValueError("Split must produce non-empty train and validation sets.")
        logger.info(
            "Split %d examples into %d train and %d validation examples.",
            len(examples),
            len(train_examples),
            len(validation_examples),
        )
        return train_examples, validation_examples

    def save_examples_jsonl(
        self, examples: Sequence[SpacyTrainingExample], output_path: Path
    ) -> str:
        """Save spaCy training examples as JSONL.

        Args:
            examples: Examples to persist.
            output_path: Destination JSONL path.

        Returns:
            String path to the saved file.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            json.dumps(
                {
                    "document_id": example.document_id,
                    "text": example.text,
                    "entities": [
                        [start, end, label] for start, end, label in example.entities
                    ],
                },
                ensure_ascii=True,
            )
            for example in examples
        ]
        output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        logger.info("Saved %d spaCy examples to %s.", len(examples), output_path)
        return str(output_path)

    def evaluate_predictions(
        self,
        gold_examples: Sequence[SpacyTrainingExample],
        predicted_examples: Sequence[SpacyTrainingExample],
    ) -> NERMetricsReport:
        """Compute entity-level precision, recall, and F1.

        Args:
            gold_examples: Ground-truth labeled examples.
            predicted_examples: Model-predicted labeled examples.

        Returns:
            Computed aggregate and per-label NER metrics.

        Raises:
            ValueError: If the example sets are mismatched by document ID.
        """
        gold_map = {example.document_id: example for example in gold_examples}
        predicted_map = {example.document_id: example for example in predicted_examples}
        if set(gold_map) != set(predicted_map):
            raise ValueError("Gold and predicted examples must contain the same document IDs.")

        overall_tp = 0
        overall_fp = 0
        overall_fn = 0
        label_counts: Dict[str, Dict[str, int]] = {}
        for document_id, gold_example in gold_map.items():
            predicted_example = predicted_map[document_id]
            gold_entities: Set[Tuple[int, int, str]] = set(gold_example.entities)
            predicted_entities: Set[Tuple[int, int, str]] = set(predicted_example.entities)
            true_positives = gold_entities & predicted_entities
            false_positives = predicted_entities - gold_entities
            false_negatives = gold_entities - predicted_entities
            overall_tp += len(true_positives)
            overall_fp += len(false_positives)
            overall_fn += len(false_negatives)

            for _, _, label in true_positives:
                self._increment_label_count(label_counts, label, "tp")
            for _, _, label in false_positives:
                self._increment_label_count(label_counts, label, "fp")
            for _, _, label in false_negatives:
                self._increment_label_count(label_counts, label, "fn")

        overall_metrics = self._build_metrics(overall_tp, overall_fp, overall_fn)
        per_label = {
            label: self._build_metrics(counts.get("tp", 0), counts.get("fp", 0), counts.get("fn", 0))
            for label, counts in sorted(label_counts.items())
        }
        report = NERMetricsReport(overall=overall_metrics, per_label=per_label)
        logger.info(
            "Evaluated NER predictions: precision=%.3f recall=%.3f f1=%.3f",
            report.overall.precision,
            report.overall.recall,
            report.overall.f1_score,
        )
        return report

    def validate_examples(self, examples: Sequence[SpacyTrainingExample]) -> List[str]:
        """Validate entity offsets and overlap constraints.

        Args:
            examples: Training examples to validate.

        Returns:
            A list of validation error messages. Empty means valid.
        """
        errors: List[str] = []
        for example in examples:
            previous_end = -1
            for start, end, label in sorted(example.entities, key=lambda item: item[0]):
                if start < 0 or end > len(example.text) or start >= end:
                    errors.append(
                        f"{example.document_id}: invalid entity offsets ({start}, {end}, {label})"
                    )
                    continue
                if start < previous_end:
                    errors.append(
                        f"{example.document_id}: overlapping entity starts at {start} before {previous_end}"
                    )
                previous_end = max(previous_end, end)
                if not label.strip():
                    errors.append(f"{example.document_id}: empty label for entity ({start}, {end})")
        if errors:
            logger.warning("Validation found %d NER example issues.", len(errors))
        else:
            logger.info("Validated %d NER examples with no errors.", len(examples))
        return errors

    def train_from_jsonl(
        self,
        input_path: Path,
        output_dir: Optional[Path] = None,
        iterations: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Train a spaCy NER model from JSONL examples.

        Args:
            input_path: spaCy JSONL training file path.
            output_dir: Optional output directory override.
            iterations: Optional training iteration override.

        Returns:
            Metadata describing the training run.

        Raises:
            NERTrainingError: If validation fails or spaCy is unavailable.
        """
        examples = self.load_examples(input_path)
        validation_errors = self.validate_examples(examples)
        if validation_errors:
            raise NERTrainingError(
                "Cannot train NER model because training data is invalid."
            )

        resolved_output_dir = output_dir or Path(self.settings.ner_model_output_dir)
        resolved_iterations = iterations or self.settings.ner_train_iterations
        spacy_module = self._import_spacy()
        model = self._build_model(spacy_module, examples)
        losses = self._train_model(spacy_module, model, examples, resolved_iterations)
        resolved_output_dir.mkdir(parents=True, exist_ok=True)
        model.to_disk(resolved_output_dir)

        metadata = {
            "examples": len(examples),
            "labels": sorted({label for example in examples for _, _, label in example.entities}),
            "iterations": resolved_iterations,
            "output_dir": str(resolved_output_dir),
            "losses": losses,
        }
        metadata_path = resolved_output_dir / "training_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        logger.info("Saved trained NER model metadata to %s.", metadata_path)
        return metadata

    def _import_spacy(self) -> Any:
        """Import spaCy lazily so data prep works without the dependency."""
        try:
            return importlib.import_module("spacy")
        except ImportError as exc:
            logger.error("spaCy is required for NER training but is not installed.")
            raise NERTrainingError(
                "spaCy is required for NER training. Install it before running ml-train."
            ) from exc

    def _build_model(self, spacy_module: Any, examples: Sequence[SpacyTrainingExample]) -> Any:
        """Create a blank spaCy pipeline and register NER labels."""
        nlp = spacy_module.blank("en")
        ner = nlp.add_pipe("ner")
        for example in examples:
            for _, _, label in example.entities:
                ner.add_label(label)
        logger.info("Registered %d unique NER labels.", len(ner.labels))
        return nlp

    def _train_model(
        self,
        spacy_module: Any,
        nlp: Any,
        examples: Sequence[SpacyTrainingExample],
        iterations: int,
    ) -> List[Dict[str, float]]:
        """Train the spaCy pipeline over the provided examples."""
        optimizer = nlp.initialize()
        losses_history: List[Dict[str, float]] = []
        for iteration in range(iterations):
            losses: Dict[str, float] = {}
            for example in examples:
                doc = nlp.make_doc(example.text)
                training_example = spacy_module.training.Example.from_dict(
                    doc,
                    {"entities": list(example.entities)},
                )
                nlp.update([training_example], sgd=optimizer, losses=losses)
            losses_history.append({"iteration": float(iteration + 1), **losses})
        logger.info("Completed %d spaCy training iterations.", iterations)
        return losses_history

    @staticmethod
    def _build_metrics(
        true_positives: int,
        false_positives: int,
        false_negatives: int,
    ) -> NERMetrics:
        """Create a metrics object from confusion counts."""
        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives)
            else 0.0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives)
            else 0.0
        )
        f1_score = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        return NERMetrics(
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
        )

    @staticmethod
    def _increment_label_count(
        label_counts: Dict[str, Dict[str, int]], label: str, key: str
    ) -> None:
        """Increment a label-specific confusion counter."""
        label_counts.setdefault(label, {"tp": 0, "fp": 0, "fn": 0})
        label_counts[label][key] += 1
