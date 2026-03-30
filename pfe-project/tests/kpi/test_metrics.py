"""Tests for KPI summary generation."""

from typing import List

from app.kpi import PipelineKPIService
from app.ml.ner_extractor import ExtractedEntity, ExtractionResult
from app.pipeline.batch_processor import PipelineBatchProcessor
from app.pipeline.decision_engine import SequentialExtractionDecisionEngine


class _FakeExtractor:
    """Minimal extractor stub for KPI tests."""

    def __init__(self, entity_sets: List[List[ExtractedEntity]]) -> None:
        self._entity_sets = entity_sets
        self._index = 0

    def extract(self, text: str) -> ExtractionResult:
        entities = self._entity_sets[self._index]
        self._index += 1
        return ExtractionResult(text=text, entities=entities)


def test_kpi_service_builds_field_and_document_metrics() -> None:
    """Ensure KPI summaries reflect batch pipeline decisions."""
    extractor = _FakeExtractor(
        [
            [ExtractedEntity(0, 5, "INV-1", "INVOICE_ID", ("regex", "spacy"), 0.95)],
            [ExtractedEntity(0, 10, "Acme Corp", "VENDOR_NAME", ("regex",), 0.89)],
        ]
    )
    processor = PipelineBatchProcessor(engine=SequentialExtractionDecisionEngine(extractor=extractor))
    batch_result = processor.run_texts(["doc one", "doc two"], ["a", "b"])

    report = PipelineKPIService().build_report(batch_result)

    assert report.document_count == 2
    assert report.auto_documents == 2
    assert report.average_field_confidence > 0.9
    assert report.field_kpis[0].total_occurrences == 1
