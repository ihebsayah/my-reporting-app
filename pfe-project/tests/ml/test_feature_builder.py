"""Tests for EntityFeatureBuilder.

Covers hand-crafted feature correctness, batch matrix shape, BERT fallback,
and the feature_dimension helper.  BERT is disabled in all tests (use_bert=False)
so the suite runs without sentence_transformers installed.
"""

from typing import Tuple

import numpy as np
import pytest

from app.ml.feature_builder import (
    _HAND_CRAFTED_DIM,
    EntityFeatureBuilder,
    FeatureVector,
)
from app.ml.ner_extractor import ExtractedEntity


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

DOC = "Invoice INV-2026-001 dated 2026-01-15. Vendor: Acme SARL. Total: $1,200.00"


def _make_entity(
    label: str = "INVOICE_ID",
    start: int = 8,
    end: int = 20,
    text: str = "INV-2026-001",
    sources: Tuple[str, ...] = ("regex",),
    score: float = 0.85,
) -> ExtractedEntity:
    return ExtractedEntity(
        start=start,
        end=end,
        text=text,
        label=label,
        sources=sources,
        score=score,
    )


@pytest.fixture()
def builder() -> EntityFeatureBuilder:
    """Return a builder with BERT disabled for fast tests."""
    return EntityFeatureBuilder(use_bert=False)


# ──────────────────────────────────────────────────────────────────────────────
# FeatureVector shape and type
# ──────────────────────────────────────────────────────────────────────────────


def test_build_returns_feature_vector(builder: EntityFeatureBuilder) -> None:
    """build() must return a FeatureVector with correct label and dtype."""
    entity = _make_entity()
    fv = builder.build(entity, DOC)

    assert isinstance(fv, FeatureVector)
    assert fv.entity_label == "INVOICE_ID"
    assert fv.features.dtype == np.float32
    assert not fv.bert_available


def test_feature_vector_length_without_bert(builder: EntityFeatureBuilder) -> None:
    """Without BERT, feature length must equal _HAND_CRAFTED_DIM."""
    fv = builder.build(_make_entity(), DOC)
    assert len(fv.features) == _HAND_CRAFTED_DIM
    assert len(fv.feature_names) == _HAND_CRAFTED_DIM


def test_feature_names_are_unique(builder: EntityFeatureBuilder) -> None:
    """All feature names must be distinct."""
    fv = builder.build(_make_entity(), DOC)
    assert len(fv.feature_names) == len(set(fv.feature_names))


# ──────────────────────────────────────────────────────────────────────────────
# Individual hand-crafted signal correctness
# ──────────────────────────────────────────────────────────────────────────────


def test_base_score_feature(builder: EntityFeatureBuilder) -> None:
    """Feature index 0 must equal entity.score."""
    entity = _make_entity(score=0.92)
    fv = builder.build(entity, DOC)
    assert fv.features[0] == pytest.approx(0.92, abs=1e-4)


def test_source_regex_flag(builder: EntityFeatureBuilder) -> None:
    """source_regex (index 1) is 1 when regex is in sources."""
    entity = _make_entity(sources=("regex",))
    fv = builder.build(entity, DOC)
    assert fv.features[1] == pytest.approx(1.0)


def test_source_spacy_flag(builder: EntityFeatureBuilder) -> None:
    """source_spacy (index 2) is 1 when spacy is in sources."""
    entity = _make_entity(sources=("spacy",))
    fv = builder.build(entity, DOC)
    assert fv.features[2] == pytest.approx(1.0)


def test_multi_source_flag(builder: EntityFeatureBuilder) -> None:
    """multi_source (index 3) is 1 when both regex and spacy are present."""
    entity = _make_entity(sources=("regex", "spacy"))
    fv = builder.build(entity, DOC)
    assert fv.features[3] == pytest.approx(1.0)


def test_multi_source_flag_off_for_single_source(builder: EntityFeatureBuilder) -> None:
    """multi_source (index 3) is 0 for a single-source entity."""
    entity = _make_entity(sources=("regex",))
    fv = builder.build(entity, DOC)
    assert fv.features[3] == pytest.approx(0.0)


def test_value_length_clipped_at_100(builder: EntityFeatureBuilder) -> None:
    """value_length (index 4) is capped at 100 characters."""
    long_text = "X" * 200
    entity = _make_entity(text=long_text, start=0, end=200)
    fv = builder.build(entity, long_text)
    assert fv.features[4] == pytest.approx(100.0)


def test_value_is_numeric_flag(builder: EntityFeatureBuilder) -> None:
    """value_is_numeric (index 5) is 1 for a pure number string."""
    entity = _make_entity(text="1200", start=64, end=68, label="TOTAL_AMOUNT")
    fv = builder.build(entity, DOC)
    assert fv.features[5] == pytest.approx(1.0)


def test_short_value_flag(builder: EntityFeatureBuilder) -> None:
    """short_value_flag (index 11) is 1 when extracted text is shorter than 3 chars."""
    entity = _make_entity(text="AB", start=0, end=2)
    fv = builder.build(entity, DOC)
    assert fv.features[11] == pytest.approx(1.0)


def test_span_position_ratio_is_normalized(builder: EntityFeatureBuilder) -> None:
    """span_position_ratio (index 8) must be in [0, 1]."""
    entity = _make_entity()
    fv = builder.build(entity, DOC)
    assert 0.0 <= float(fv.features[8]) <= 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Batch matrix
# ──────────────────────────────────────────────────────────────────────────────


def test_build_batch_returns_matrix(builder: EntityFeatureBuilder) -> None:
    """build_batch must return a 2-D matrix with one row per entity."""
    entities = [
        _make_entity(label="INVOICE_ID"),
        _make_entity(label="INVOICE_DATE", start=27, end=37, text="2026-01-15"),
    ]
    matrix, names = builder.build_batch(entities, DOC)

    assert matrix.shape == (2, _HAND_CRAFTED_DIM)
    assert len(names) == _HAND_CRAFTED_DIM


def test_build_batch_empty_raises(builder: EntityFeatureBuilder) -> None:
    """build_batch with an empty entity list must raise ValueError."""
    with pytest.raises(ValueError, match="empty"):
        builder.build_batch([], DOC)


# ──────────────────────────────────────────────────────────────────────────────
# feature_dimension helper
# ──────────────────────────────────────────────────────────────────────────────


def test_feature_dimension_without_bert(builder: EntityFeatureBuilder) -> None:
    """feature_dimension must equal _HAND_CRAFTED_DIM when BERT is off."""
    assert builder.feature_dimension() == _HAND_CRAFTED_DIM


def test_feature_dimension_matches_built_vector(builder: EntityFeatureBuilder) -> None:
    """feature_dimension must match the actual built vector length."""
    fv = builder.build(_make_entity(), DOC)
    assert builder.feature_dimension() == len(fv.features)
