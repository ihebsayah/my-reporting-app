"""Machine learning package."""

from app.ml.confidence_scorer import ConfidenceAssessment, FieldConfidenceScorer
from app.ml.feature_builder import EntityFeatureBuilder, FeatureVector
from app.ml.ner_extractor import (
    ExtractedEntity,
    ExtractionResult,
    RegexSpacyEnsembleExtractor,
)
from app.ml.ner_trainer import (
    NERMetrics,
    NERTrainingError,
    SpacyNERTrainer,
    SpacyTrainingExample,
)
from app.ml.rf_confidence_model import (
    ConfidencePrediction,
    RFConfidenceModel,
    RFModelNotLoadedError,
    RFModelTrainingError,
    RFTrainingResult,
    TrainingRecord,
)

__all__ = [
    "ConfidenceAssessment",
    "ConfidencePrediction",
    "EntityFeatureBuilder",
    "ExtractedEntity",
    "ExtractionResult",
    "FeatureVector",
    "FieldConfidenceScorer",
    "NERMetrics",
    "NERTrainingError",
    "RFConfidenceModel",
    "RFModelNotLoadedError",
    "RFModelTrainingError",
    "RFTrainingResult",
    "RegexSpacyEnsembleExtractor",
    "SpacyNERTrainer",
    "SpacyTrainingExample",
    "TrainingRecord",
]
