"""Tests for the regex + spaCy ensemble extractor."""

from typing import Any, List

from app.ml.ner_extractor import RegexSpacyEnsembleExtractor


class _FakeEntity:
    """Minimal fake spaCy entity for tests."""

    def __init__(self, start_char: int, end_char: int, text: str, label_: str) -> None:
        self.start_char = start_char
        self.end_char = end_char
        self.text = text
        self.label_ = label_


class _FakeDoc:
    """Minimal fake spaCy doc for tests."""

    def __init__(self, ents: List[_FakeEntity]) -> None:
        self.ents = ents


class _FakeSpacyModel:
    """Minimal fake spaCy pipeline for tests."""

    def __init__(self, ents: List[_FakeEntity]) -> None:
        self._ents = ents

    def __call__(self, text: str) -> _FakeDoc:
        return _FakeDoc(self._ents)


def test_regex_extractor_finds_invoice_fields() -> None:
    """Ensure regex extraction finds core invoice entities."""
    text = "Invoice INV-2026-001\nVendor: Acme Supplies LLC\nDate: 2026-03-29\nTotal: $12.00"
    extractor = RegexSpacyEnsembleExtractor(spacy_model=None)

    result = extractor.extract(text)
    labels = [entity.label for entity in result.entities]

    assert "INVOICE_ID" in labels
    assert "VENDOR_NAME" in labels
    assert "INVOICE_DATE" in labels
    assert "TOTAL_AMOUNT" in labels


def test_extractor_falls_back_to_regex_only_when_spacy_model_unavailable(monkeypatch: Any) -> None:
    """Ensure extraction still works when spaCy loading is unavailable."""
    text = "Invoice INV-2026-001\nTotal: $12.00"
    extractor = RegexSpacyEnsembleExtractor(spacy_model=None)
    monkeypatch.setattr(extractor, "_load_spacy_model", lambda: None)

    result = extractor.extract(text)

    assert result.entities
    assert all(entity.sources == ("regex",) for entity in result.entities)


def test_ensemble_merges_exact_span_matches_from_regex_and_spacy() -> None:
    """Ensure matching regex and spaCy entities are merged with both sources."""
    text = "Invoice INV-2026-001\nTotal: $12.00"
    fake_model = _FakeSpacyModel(
        [
            _FakeEntity(8, 20, "INV-2026-001", "INVOICE_ID"),
            _FakeEntity(28, 34, "$12.00", "TOTAL_AMOUNT"),
        ]
    )
    extractor = RegexSpacyEnsembleExtractor(spacy_model=fake_model)

    result = extractor.extract(text)
    invoice_entity = next(entity for entity in result.entities if entity.label == "INVOICE_ID")

    assert "regex" in invoice_entity.sources
    assert "spacy" in invoice_entity.sources
    assert invoice_entity.score > 0.85
