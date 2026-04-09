"""Feature engineering for extraction confidence scoring.

Produces a fixed-length numeric feature vector for each extracted entity.
The vector combines hand-crafted signals (character patterns, source flags,
length ratios) with an optional BERT sentence embedding so the Random Forest
classifier can learn both rule-based and semantic cues.

Architecture decision #3: hand-crafted + BERT embeddings (BERT is optional;
the pipeline degrades gracefully to hand-crafted features if the transformer
library is not installed).
"""

import importlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.ml.ner_extractor import ExtractedEntity

logger = logging.getLogger(__name__)

# Number of hand-crafted scalar features produced by _hand_crafted_vector.
_HAND_CRAFTED_DIM = 12
# BERT sentence-transformer embedding dimension (all-MiniLM-L6-v2).
_BERT_DIM = 384


@dataclass(frozen=True)
class FeatureVector:
    """Numeric feature vector for a single extracted entity.

    Attributes:
        entity_label: The NER label (e.g. ``INVOICE_ID``).
        features: 1-D numpy float32 array of length
            ``_HAND_CRAFTED_DIM`` (no BERT) or
            ``_HAND_CRAFTED_DIM + _BERT_DIM`` (with BERT).
        feature_names: Ordered list of feature names aligned with ``features``.
        bert_available: Whether BERT embeddings were included.
    """

    entity_label: str
    features: np.ndarray
    feature_names: List[str]
    bert_available: bool


@dataclass
class EntityFeatureBuilder:
    """Build feature vectors for extracted entities.

    Combines twelve hand-crafted scalar signals with an optional
    384-dimensional BERT sentence embedding.  The BERT model is loaded
    lazily and cached after the first call so repeated batch scoring
    has no warm-up overhead.

    Args:
        bert_model_name: Sentence-transformer model to load when BERT is
            available.  Defaults to ``all-MiniLM-L6-v2``.
        use_bert: Set ``False`` to disable BERT even when the library is
            installed (useful in unit tests for speed).
    """

    bert_model_name: str = "all-MiniLM-L6-v2"
    use_bert: bool = True
    _bert_encoder: Any = field(default=None, init=False, repr=False, compare=False)

    def build(self, entity: ExtractedEntity, document_text: str) -> FeatureVector:
        """Build a feature vector for one extracted entity.

        Args:
            entity: The entity to featurise.
            document_text: The full source document text (used for context
                features such as span position ratio and surrounding chars).

        Returns:
            A ``FeatureVector`` with hand-crafted features and, when
            available, concatenated BERT embeddings.
        """
        hand_crafted, hc_names = self._hand_crafted_vector(entity, document_text)
        bert_embedding, bert_names, bert_ok = self._bert_vector(entity.text)

        if bert_ok:
            features = np.concatenate([hand_crafted, bert_embedding]).astype(np.float32)
            feature_names = hc_names + bert_names
        else:
            features = hand_crafted.astype(np.float32)
            feature_names = hc_names

        logger.debug(
            "Built %d-dim feature vector for label=%s bert=%s.",
            len(features),
            entity.label,
            bert_ok,
        )
        return FeatureVector(
            entity_label=entity.label,
            features=features,
            feature_names=feature_names,
            bert_available=bert_ok,
        )

    def build_batch(
        self,
        entities: List[ExtractedEntity],
        document_text: str,
    ) -> Tuple[np.ndarray, List[str]]:
        """Build a 2-D feature matrix for a list of entities.

        Args:
            entities: Extracted entities to featurise.
            document_text: The full source document text.

        Returns:
            Tuple of (matrix of shape ``[n_entities, n_features]``,
            ordered feature name list).

        Raises:
            ValueError: If ``entities`` is empty.
        """
        if not entities:
            raise ValueError("Cannot build a feature matrix from an empty entity list.")

        vectors = [self.build(entity, document_text) for entity in entities]
        matrix = np.stack([v.features for v in vectors], axis=0)
        logger.info(
            "Built feature matrix %s for %d entities.",
            matrix.shape,
            len(entities),
        )
        return matrix, vectors[0].feature_names

    # ------------------------------------------------------------------
    # Hand-crafted features
    # ------------------------------------------------------------------

    def _hand_crafted_vector(
        self, entity: ExtractedEntity, document_text: str
    ) -> Tuple[np.ndarray, List[str]]:
        """Compute twelve deterministic scalar features.

        Features
        --------
        0  base_score          – raw extraction score (0.0–1.0)
        1  source_regex        – 1 if 'regex' in sources else 0
        2  source_spacy        – 1 if 'spacy' in sources else 0
        3  multi_source        – 1 if both sources present else 0
        4  value_length        – char count of extracted text (clipped to 100)
        5  value_is_numeric    – 1 if extracted text is purely numeric
        6  value_has_alpha     – 1 if extracted text contains letters
        7  value_has_special   – 1 if text has -, /, $, €, £, %
        8  span_position_ratio – start_char / document_length
        9  char_density        – value chars / span char range
        10 all_caps            – 1 if value is UPPER CASE
        11 short_value_flag    – 1 if value length < 3 (suspicious)
        """
        text_val = entity.text
        doc_len = max(len(document_text), 1)
        span_len = max(entity.end - entity.start, 1)

        features = np.array(
            [
                float(entity.score),
                1.0 if "regex" in entity.sources else 0.0,
                1.0 if "spacy" in entity.sources else 0.0,
                1.0 if len(entity.sources) > 1 else 0.0,
                min(float(len(text_val)), 100.0),
                1.0 if re.fullmatch(r"[\d,. ]+", text_val) else 0.0,
                1.0 if re.search(r"[A-Za-z]", text_val) else 0.0,
                1.0 if re.search(r"[-/$€£%]", text_val) else 0.0,
                float(entity.start) / doc_len,
                float(len(text_val)) / span_len,
                1.0 if text_val == text_val.upper() and text_val.strip() else 0.0,
                1.0 if len(text_val.strip()) < 3 else 0.0,
            ],
            dtype=np.float64,
        )
        names = [
            "base_score",
            "source_regex",
            "source_spacy",
            "multi_source",
            "value_length",
            "value_is_numeric",
            "value_has_alpha",
            "value_has_special",
            "span_position_ratio",
            "char_density",
            "all_caps",
            "short_value_flag",
        ]
        assert len(features) == _HAND_CRAFTED_DIM, "Hand-crafted feature count mismatch."
        return features, names

    # ------------------------------------------------------------------
    # BERT embedding features
    # ------------------------------------------------------------------

    def _bert_vector(
        self, text: str
    ) -> Tuple[np.ndarray, List[str], bool]:
        """Encode text with a sentence-transformer model.

        Falls back to a zero vector when the library is not installed or
        ``use_bert`` is ``False``.

        Args:
            text: The entity value to encode.

        Returns:
            Tuple of (embedding array, feature names, bert_was_used).
        """
        bert_names = [f"bert_{i}" for i in range(_BERT_DIM)]

        if not self.use_bert:
            return np.zeros(_BERT_DIM, dtype=np.float64), bert_names, False

        encoder = self._load_bert_encoder()
        if encoder is None:
            return np.zeros(_BERT_DIM, dtype=np.float64), bert_names, False

        try:
            embedding = encoder.encode(text, show_progress_bar=False)
            return np.array(embedding, dtype=np.float64), bert_names, True
        except Exception:
            logger.warning(
                "BERT encoding failed for text=%r; falling back to zeros.", text[:40]
            )
            return np.zeros(_BERT_DIM, dtype=np.float64), bert_names, False

    def _load_bert_encoder(self) -> Optional[Any]:
        """Load the sentence-transformer model lazily and cache it.

        Returns:
            The encoder object, or ``None`` if unavailable.
        """
        if self._bert_encoder is not None:
            return self._bert_encoder

        try:
            st_module = importlib.import_module("sentence_transformers")
            self._bert_encoder = st_module.SentenceTransformer(self.bert_model_name)
            logger.info("Loaded BERT encoder: %s.", self.bert_model_name)
            return self._bert_encoder
        except ImportError:
            logger.info(
                "sentence_transformers not installed; using hand-crafted features only."
            )
            return None
        except Exception as exc:
            logger.warning("Failed to load BERT encoder: %s.", exc)
            return None

    def feature_dimension(self) -> int:
        """Return the total feature vector length.

        Returns:
            ``_HAND_CRAFTED_DIM + _BERT_DIM`` if BERT will be used,
            ``_HAND_CRAFTED_DIM`` otherwise.
        """
        if not self.use_bert:
            return _HAND_CRAFTED_DIM
        encoder = self._load_bert_encoder()
        return _HAND_CRAFTED_DIM + _BERT_DIM if encoder is not None else _HAND_CRAFTED_DIM
