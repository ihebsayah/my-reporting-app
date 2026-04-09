"""Random Forest confidence model for extraction quality scoring.

Architecture decision #4: the confidence model is a Random Forest
classifier trained on hand-crafted + BERT features (see feature_builder.py).
Target accuracy: >= 85%.

The model predicts a continuous confidence score [0.0, 1.0] for each
extracted entity -- derived from the RF's ``predict_proba`` output --
and routes each field to one of three decisions:

  * ``auto``   – confidence >= per-field auto threshold (default 0.90+)
  * ``review`` – confidence in [review_min, review_max]  (default 0.70-0.90)
  * ``reject`` – confidence < review_min                 (default < 0.70)

Architecture decision #8: thresholds are per-field, loaded from
``Settings.field_thresholds()``.

Versioning follows the project convention:
  ``rf_confidence_v{YYYYMMDD}_{HHMM}.joblib``
"""

import importlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from app.config import Settings, get_settings
from app.ml.feature_builder import EntityFeatureBuilder
from app.ml.ner_extractor import ExtractedEntity

logger = logging.getLogger(__name__)

# Label used for the "correct extraction" positive class during training.
_POSITIVE_CLASS = 1
_NEGATIVE_CLASS = 0


# ──────────────────────────────────────────────────────────────────────────────
# Data-transfer objects
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TrainingRecord:
    """One labeled example for RF confidence model training.

    Attributes:
        entity: Extracted entity to score.
        document_text: Full document text the entity came from.
        is_correct: True if a human annotator confirmed the extraction.
    """

    entity: ExtractedEntity
    document_text: str
    is_correct: bool


@dataclass(frozen=True)
class ConfidencePrediction:
    """Confidence prediction for a single extracted entity.

    Attributes:
        entity_label: The NER label (e.g. ``INVOICE_ID``).
        entity_text: The extracted value string.
        confidence: Predicted confidence score in [0.0, 1.0].
        decision: One of ``auto``, ``review``, or ``reject``.
        model_version: Identifier of the model that produced this prediction.
    """

    entity_label: str
    entity_text: str
    confidence: float
    decision: str
    model_version: str


@dataclass(frozen=True)
class RFTrainingResult:
    """Summary of a completed RF training run.

    Attributes:
        model_version: Versioned filename stem.
        model_path: Absolute path to the saved ``.joblib`` file.
        train_samples: Number of training samples used.
        accuracy: Training-set accuracy (not a substitute for validation).
        feature_count: Dimension of the input feature vectors.
        labels: List of NER labels seen during training.
        trained_at: ISO-8601 UTC timestamp.
    """

    model_version: str
    model_path: str
    train_samples: int
    accuracy: float
    feature_count: int
    labels: List[str]
    trained_at: str


# ──────────────────────────────────────────────────────────────────────────────
# Exceptions
# ──────────────────────────────────────────────────────────────────────────────


class RFModelNotLoadedError(RuntimeError):
    """Raised when prediction is attempted without a loaded model."""


class RFModelTrainingError(RuntimeError):
    """Raised when RF model training fails."""


# ──────────────────────────────────────────────────────────────────────────────
# Main class
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class RFConfidenceModel:
    """Random Forest classifier for extraction confidence scoring.

    Usage
    -----
    Training (Month 2, after annotation export)::

        from app.ml.rf_confidence_model import RFConfidenceModel, TrainingRecord
        model = RFConfidenceModel()
        records = [TrainingRecord(entity=…, document_text=…, is_correct=True), …]
        result = model.train(records)
        print(result.accuracy)

    Prediction (pipeline integration, Month 3)::

        model = RFConfidenceModel()
        model.load(Path("artifacts/models/rf_confidence/rf_confidence_v20260408_1530.joblib"))
        prediction = model.predict(entity, document_text)
        print(prediction.confidence, prediction.decision)

    Args:
        settings: Optional settings override (useful in tests).
        feature_builder: Optional feature builder override.
        use_bert: Passed to the default ``EntityFeatureBuilder``; set
            ``False`` in unit tests for speed.
    """

    settings: Any = field(default=None)
    feature_builder: Optional[EntityFeatureBuilder] = field(default=None)
    use_bert: bool = field(default=True)

    _model: Any = field(default=None, init=False, repr=False, compare=False)
    _model_version: str = field(default="unloaded", init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Resolve defaults after dataclass construction."""
        if self.settings is None:
            self.settings = get_settings()
        if self.feature_builder is None:
            self.feature_builder = EntityFeatureBuilder(use_bert=self.use_bert)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        records: Sequence[TrainingRecord],
        output_dir: Optional[Path] = None,
        n_estimators: int = 200,
        random_state: int = 42,
    ) -> RFTrainingResult:
        """Train a Random Forest model and save it to disk.

        Args:
            records: Labeled training records.
            output_dir: Directory to save the model; defaults to
                ``Settings.rf_model_output_dir``.
            n_estimators: Number of trees in the forest.
            random_state: Seed for deterministic training.

        Returns:
            A ``RFTrainingResult`` summary with path, accuracy, and metadata.

        Raises:
            RFModelTrainingError: If scikit-learn is missing or the feature
                matrix cannot be built.
            ValueError: If ``records`` is empty or has fewer than 2 samples.
        """
        if len(records) < 2:
            raise ValueError("At least 2 labeled records are required for RF training.")

        sklearn = self._import_sklearn()
        resolved_dir = output_dir or Path(self.settings.rf_model_output_dir)

        logger.info("Building feature matrix for %d training records.", len(records))
        try:
            X, feature_names = self._build_matrix(records)
            y = np.array(
                [_POSITIVE_CLASS if r.is_correct else _NEGATIVE_CLASS for r in records],
                dtype=np.int32,
            )
        except Exception as exc:
            logger.error("Feature matrix construction failed: %s.", exc)
            raise RFModelTrainingError("Feature extraction failed during RF training.") from exc

        rf = sklearn.ensemble.RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        rf.fit(X, y)
        accuracy = float(rf.score(X, y))
        logger.info(
            "RF training complete: accuracy=%.4f samples=%d features=%d.",
            accuracy,
            len(records),
            X.shape[1],
        )

        version = self._make_version_tag()
        saved_path = self._save_model(rf, resolved_dir, version)

        result = RFTrainingResult(
            model_version=version,
            model_path=str(saved_path),
            train_samples=len(records),
            accuracy=accuracy,
            feature_count=X.shape[1],
            labels=sorted({r.entity.label for r in records}),
            trained_at=datetime.now(timezone.utc).isoformat(),
        )
        metadata_path = saved_path.with_suffix(".json")
        metadata_path.write_text(
            json.dumps(
                {
                    "model_version": result.model_version,
                    "train_samples": result.train_samples,
                    "accuracy": result.accuracy,
                    "feature_count": result.feature_count,
                    "labels": result.labels,
                    "trained_at": result.trained_at,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("Saved RF model metadata to %s.", metadata_path)

        # Cache the freshly trained model.
        self._model = rf
        self._model_version = version
        return result

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self, entity: ExtractedEntity, document_text: str
    ) -> ConfidencePrediction:
        """Predict confidence for a single extracted entity.

        Args:
            entity: The extracted entity to score.
            document_text: Full source document text.

        Returns:
            A ``ConfidencePrediction`` with score and routing decision.

        Raises:
            RFModelNotLoadedError: If no model has been loaded or trained yet.
        """
        if self._model is None:
            raise RFModelNotLoadedError(
                "No RF model loaded. Call train() or load() first."
            )

        feature_vec = self.feature_builder.build(entity, document_text)
        X = feature_vec.features.reshape(1, -1)
        proba = self._model.predict_proba(X)

        # Find the probability of the positive class.
        classes = list(self._model.classes_)
        pos_index = classes.index(_POSITIVE_CLASS) if _POSITIVE_CLASS in classes else -1
        confidence = float(proba[0, pos_index]) if pos_index >= 0 else 0.5

        decision = self._route_decision(entity.label, confidence)
        logger.debug(
            "RF confidence for label=%s text=%r → %.3f (%s).",
            entity.label,
            entity.text[:30],
            confidence,
            decision,
        )
        return ConfidencePrediction(
            entity_label=entity.label,
            entity_text=entity.text,
            confidence=confidence,
            decision=decision,
            model_version=self._model_version,
        )

    def predict_batch(
        self,
        entities: List[ExtractedEntity],
        document_text: str,
    ) -> List[ConfidencePrediction]:
        """Predict confidence for a list of entities in one pass.

        Args:
            entities: Extracted entities to score.
            document_text: Full source document text.

        Returns:
            List of ``ConfidencePrediction`` objects, one per entity.

        Raises:
            RFModelNotLoadedError: If no model is loaded.
            ValueError: If ``entities`` is empty.
        """
        if self._model is None:
            raise RFModelNotLoadedError(
                "No RF model loaded. Call train() or load() first."
            )
        if not entities:
            raise ValueError("entities list must not be empty.")

        X, _ = self.feature_builder.build_batch(entities, document_text)
        proba = self._model.predict_proba(X)
        classes = list(self._model.classes_)
        pos_index = classes.index(_POSITIVE_CLASS) if _POSITIVE_CLASS in classes else -1

        predictions: List[ConfidencePrediction] = []
        for i, entity in enumerate(entities):
            confidence = float(proba[i, pos_index]) if pos_index >= 0 else 0.5
            decision = self._route_decision(entity.label, confidence)
            predictions.append(
                ConfidencePrediction(
                    entity_label=entity.label,
                    entity_text=entity.text,
                    confidence=confidence,
                    decision=decision,
                    model_version=self._model_version,
                )
            )

        logger.info(
            "Batch RF prediction complete for %d entities.", len(entities)
        )
        return predictions

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self, model_path: Path) -> None:
        """Load a previously saved RF model from disk.

        Args:
            model_path: Path to a ``.joblib`` file saved by ``train()``.

        Raises:
            FileNotFoundError: If the path does not exist.
            RFModelTrainingError: If joblib is not installed.
        """
        if not model_path.exists():
            raise FileNotFoundError(f"RF model file not found: {model_path}")

        joblib = self._import_joblib()
        self._model = joblib.load(model_path)
        self._model_version = model_path.stem
        logger.info("Loaded RF confidence model from %s.", model_path)

    # ------------------------------------------------------------------
    # Decision routing (architecture decision #8)
    # ------------------------------------------------------------------

    def _route_decision(self, label: str, confidence: float) -> str:
        """Apply per-field threshold rules to produce a routing decision.

        Args:
            label: The NER entity label (e.g. ``INVOICE_ID``).
            confidence: Predicted confidence score in [0.0, 1.0].

        Returns:
            One of ``"auto"``, ``"review"``, or ``"reject"``.
        """
        thresholds = self._get_field_thresholds(label)
        auto_thresh = thresholds["auto"]
        review_min = thresholds["review_min"]

        if confidence >= auto_thresh:
            return "auto"
        if confidence >= review_min:
            return "review"
        return "reject"

    def _get_field_thresholds(self, label: str) -> Dict[str, float]:
        """Return thresholds for a specific label, falling back to defaults.

        Args:
            label: Entity label to look up.

        Returns:
            Dict with ``auto``, ``review_min``, ``review_max`` keys.
        """
        try:
            per_field = self.settings.field_thresholds()
        except Exception:
            per_field = {}

        return per_field.get(
            label,
            {
                "auto": float(self.settings.auto_approval_threshold),
                "review_min": float(self.settings.review_min_threshold),
                "review_max": float(self.settings.review_max_threshold),
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_matrix(
        self, records: Sequence[TrainingRecord]
    ) -> Tuple[np.ndarray, List[str]]:
        """Turn training records into a 2-D feature matrix."""
        entities = [r.entity for r in records]
        # Use the first record's document text (batch call needs one text;
        # in production, call build() individually for mixed documents).
        document_text = records[0].document_text if records else ""
        return self.feature_builder.build_batch(entities, document_text)

    def _save_model(self, model: Any, output_dir: Path, version: str) -> Path:
        """Persist the model with a versioned filename."""
        joblib = self._import_joblib()
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{version}.joblib"
        joblib.dump(model, path)
        logger.info("Saved RF model to %s.", path)
        return path

    @staticmethod
    def _make_version_tag() -> str:
        """Generate a timestamp-based version string."""
        now = datetime.now(timezone.utc)
        return f"rf_confidence_v{now.strftime('%Y%m%d_%H%M')}"

    @staticmethod
    def _import_sklearn() -> Any:
        """Import scikit-learn lazily."""
        try:
            return importlib.import_module("sklearn")
        except ImportError as exc:
            raise RFModelTrainingError(
                "scikit-learn is required for RF training. "
                "Install it with: pip install scikit-learn"
            ) from exc

    @staticmethod
    def _import_joblib() -> Any:
        """Import joblib lazily (bundled with scikit-learn)."""
        try:
            return importlib.import_module("joblib")
        except ImportError as exc:
            raise RFModelTrainingError(
                "joblib is required to save/load the RF model. "
                "Install it with: pip install joblib"
            ) from exc
