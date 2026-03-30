"""Batch pipeline processing and decision metrics."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from app.annotation.task_builder import load_documents_from_directory
from app.config import Settings, get_settings
from app.pipeline.decision_engine import PipelineDecisionResult, SequentialExtractionDecisionEngine

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BatchDocumentResult:
    """Represents one processed document in a batch run."""

    document_id: str
    result: PipelineDecisionResult


@dataclass(frozen=True)
class BatchMetrics:
    """Aggregated pipeline metrics across a batch of documents."""

    document_count: int
    overall_decisions: Dict[str, int]
    field_decisions: Dict[str, Dict[str, int]]


@dataclass(frozen=True)
class BatchProcessingResult:
    """Represents the output of a batch pipeline run."""

    documents: List[BatchDocumentResult] = field(default_factory=list)
    metrics: Optional[BatchMetrics] = None


class PipelineBatchProcessor:
    """Run the sequential pipeline across multiple documents and aggregate metrics."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        engine: Optional[SequentialExtractionDecisionEngine] = None,
    ) -> None:
        """Initialize the batch processor.

        Args:
            settings: Optional settings override.
            engine: Optional decision engine override.
        """
        self.settings = settings or get_settings()
        self.engine = engine or SequentialExtractionDecisionEngine(settings=self.settings)

    def run_texts(self, texts: Sequence[str], document_ids: Optional[Sequence[str]] = None) -> BatchProcessingResult:
        """Run the pipeline on a sequence of raw texts.

        Args:
            texts: Raw document texts.
            document_ids: Optional aligned document IDs.

        Returns:
            Batch processing result with aggregated metrics.
        """
        resolved_ids = list(document_ids) if document_ids is not None else [
            f"document-{index + 1}" for index in range(len(texts))
        ]
        document_results = [
            BatchDocumentResult(
                document_id=document_id,
                result=self.engine.run(text),
            )
            for document_id, text in zip(resolved_ids, texts)
        ]
        metrics = self._build_metrics(document_results)
        logger.info("Processed %d documents in batch mode.", len(document_results))
        return BatchProcessingResult(documents=document_results, metrics=metrics)

    def run_directory(self, input_dir: str) -> BatchProcessingResult:
        """Run the pipeline on source documents from a directory.

        Args:
            input_dir: Directory containing annotation source documents.

        Returns:
            Batch processing result with aggregated metrics.
        """
        documents = load_documents_from_directory(Path(input_dir))
        texts = [document.text for document in documents]
        document_ids = [document.document_id for document in documents]
        return self.run_texts(texts=texts, document_ids=document_ids)

    def _build_metrics(self, documents: Sequence[BatchDocumentResult]) -> BatchMetrics:
        """Aggregate overall and per-field decision metrics."""
        overall_decisions: Dict[str, int] = {"auto": 0, "review": 0, "reject": 0}
        field_decisions: Dict[str, Dict[str, int]] = {}

        for document in documents:
            overall_decisions[document.result.overall_decision] = (
                overall_decisions.get(document.result.overall_decision, 0) + 1
            )
            for field in document.result.fields:
                field_decisions.setdefault(
                    field.field_name,
                    {"auto": 0, "review": 0, "reject": 0},
                )
                field_decisions[field.field_name][field.decision] = (
                    field_decisions[field.field_name].get(field.decision, 0) + 1
                )

        return BatchMetrics(
            document_count=len(documents),
            overall_decisions=overall_decisions,
            field_decisions=field_decisions,
        )
