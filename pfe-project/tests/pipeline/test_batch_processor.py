"""Tests for batch pipeline processing and metrics."""

from typing import List

from app.ml.ner_extractor import ExtractedEntity, ExtractionResult
from app.pipeline.batch_processor import PipelineBatchProcessor
from app.pipeline.decision_engine import SequentialExtractionDecisionEngine


class _FakeExtractor:
    """Minimal extractor stub for batch tests."""

    def __init__(self, entity_sets: List[List[ExtractedEntity]]) -> None:
        self._entity_sets = entity_sets
        self._index = 0

    def extract(self, text: str) -> ExtractionResult:
        entities = self._entity_sets[self._index]
        self._index += 1
        return ExtractionResult(text=text, entities=entities)


def test_batch_processor_aggregates_overall_and_field_decisions() -> None:
    """Ensure batch processing counts overall and per-field decisions."""
    extractor = _FakeExtractor(
        [
            [ExtractedEntity(0, 5, "INV-1", "INVOICE_ID", ("regex", "spacy"), 0.95)],
            [ExtractedEntity(0, 6, "$10.00", "TOTAL_AMOUNT", ("regex",), 0.85)],
            [],
        ]
    )
    engine = SequentialExtractionDecisionEngine(extractor=extractor)
    processor = PipelineBatchProcessor(engine=engine)

    result = processor.run_texts(["doc one", "doc two", "doc three"], ["a", "b", "c"])

    assert result.metrics is not None
    assert result.metrics.document_count == 3
    assert result.metrics.overall_decisions["auto"] == 1
    assert result.metrics.overall_decisions["review"] == 1
    assert result.metrics.overall_decisions["reject"] == 1
    assert result.metrics.field_decisions["INVOICE_ID"]["auto"] == 1
    assert result.metrics.field_decisions["TOTAL_AMOUNT"]["review"] == 1
