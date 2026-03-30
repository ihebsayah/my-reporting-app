"""KPI summary builders for pipeline outputs."""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from app.pipeline.batch_processor import BatchProcessingResult
from app.pipeline.decision_engine import FieldDecision

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FieldKPI:
    """Represents KPI metrics for one extracted field."""

    field_name: str
    total_occurrences: int
    auto_count: int
    review_count: int
    reject_count: int
    average_confidence: float


@dataclass(frozen=True)
class PipelineKPIReport:
    """Represents aggregated KPI metrics for a batch pipeline run."""

    document_count: int
    auto_documents: int
    review_documents: int
    reject_documents: int
    average_field_confidence: float
    field_kpis: List[FieldKPI] = field(default_factory=list)


class PipelineKPIService:
    """Build reusable KPI summaries from batch pipeline results."""

    def build_report(self, batch_result: BatchProcessingResult) -> PipelineKPIReport:
        """Create a KPI report from a batch processing result.

        Args:
            batch_result: Batch pipeline output.

        Returns:
            Structured KPI report.
        """
        metrics = batch_result.metrics
        overall_decisions = metrics.overall_decisions if metrics else {}
        field_kpis = self._build_field_kpis(batch_result)
        average_field_confidence = self._average_confidence(batch_result)
        report = PipelineKPIReport(
            document_count=len(batch_result.documents),
            auto_documents=overall_decisions.get("auto", 0),
            review_documents=overall_decisions.get("review", 0),
            reject_documents=overall_decisions.get("reject", 0),
            average_field_confidence=average_field_confidence,
            field_kpis=field_kpis,
        )
        logger.info(
            "Built KPI report for %d documents with %.3f average field confidence.",
            report.document_count,
            report.average_field_confidence,
        )
        return report

    def _build_field_kpis(self, batch_result: BatchProcessingResult) -> List[FieldKPI]:
        """Aggregate KPI values by field."""
        field_totals: Dict[str, Dict[str, float]] = {}
        for document in batch_result.documents:
            for field in document.result.fields:
                field_totals.setdefault(
                    field.field_name,
                    {
                        "total": 0.0,
                        "auto": 0.0,
                        "review": 0.0,
                        "reject": 0.0,
                        "confidence_sum": 0.0,
                    },
                )
                stats = field_totals[field.field_name]
                stats["total"] += 1
                stats[field.decision] += 1
                stats["confidence_sum"] += field.confidence

        return [
            FieldKPI(
                field_name=field_name,
                total_occurrences=int(stats["total"]),
                auto_count=int(stats["auto"]),
                review_count=int(stats["review"]),
                reject_count=int(stats["reject"]),
                average_confidence=(
                    stats["confidence_sum"] / stats["total"] if stats["total"] else 0.0
                ),
            )
            for field_name, stats in sorted(field_totals.items())
        ]

    def _average_confidence(self, batch_result: BatchProcessingResult) -> float:
        """Calculate average confidence across all extracted fields."""
        confidences: List[float] = [
            field.confidence
            for document in batch_result.documents
            for field in document.result.fields
        ]
        return sum(confidences) / len(confidences) if confidences else 0.0


def kpi_report_to_payload(report: PipelineKPIReport) -> Dict[str, object]:
    """Serialize a KPI report for API or CLI output."""
    return {
        "document_count": report.document_count,
        "auto_documents": report.auto_documents,
        "review_documents": report.review_documents,
        "reject_documents": report.reject_documents,
        "average_field_confidence": report.average_field_confidence,
        "field_kpis": [
            {
                "field_name": field.field_name,
                "total_occurrences": field.total_occurrences,
                "auto_count": field.auto_count,
                "review_count": field.review_count,
                "reject_count": field.reject_count,
                "average_confidence": field.average_confidence,
            }
            for field in report.field_kpis
        ],
    }
