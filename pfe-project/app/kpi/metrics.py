"""KPI summary builders for pipeline outputs.

Month 3 additions
-----------------
* ``NERKPIService`` -- computes entity-level precision, recall, and F1 from
  gold-label annotations vs pipeline predictions.  Target: >= 94% accuracy
  over 100 documents (Month 3 deliverable).
* ``ExtractionKPIService`` -- derives throughput, auto-rate, and drift
  metrics from persisted ``ExtractionRecord`` rows (feeds the Streamlit
  dashboard in Month 4).

The original ``PipelineKPIService`` is unchanged so existing API tests keep
passing.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from app.pipeline.batch_processor import BatchProcessingResult
from app.pipeline.decision_engine import FieldDecision

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Existing KPI types (unchanged)
# ──────────────────────────────────────────────────────────────────────────────


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


# ──────────────────────────────────────────────────────────────────────────────
# New: NER precision / recall / F1 (Month 3)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EntitySpan:
    """Canonical span representation for KPI comparison.

    Attributes:
        document_id: Source document identifier.
        label: NER label (e.g. ``INVOICE_ID``).
        value: Extracted or gold-label text value.
    """

    document_id: str
    label: str
    value: str


@dataclass(frozen=True)
class NERFieldMetrics:
    """Precision, recall, and F1 for one NER label.

    Attributes:
        label: NER entity label.
        true_positives: Correctly extracted spans.
        false_positives: Extracted but not in gold.
        false_negatives: In gold but not extracted.
        precision: TP / (TP + FP).
        recall: TP / (TP + FN).
        f1: Harmonic mean of precision and recall.
    """

    label: str
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class NERKPIReport:
    """Aggregate and per-label NER KPI report.

    Attributes:
        document_count: Number of documents evaluated.
        overall_precision: Macro-averaged precision.
        overall_recall: Macro-averaged recall.
        overall_f1: Macro-averaged F1 score.
        per_label: Per-NER-label metrics.
        meets_target: True when overall_f1 >= ``f1_target``.
        f1_target: The target F1 threshold checked.
    """

    document_count: int
    overall_precision: float
    overall_recall: float
    overall_f1: float
    per_label: List[NERFieldMetrics]
    meets_target: bool
    f1_target: float


class NERKPIService:
    """Compute entity-level precision, recall, and F1 KPIs.

    Compares gold-label spans (from human annotation) with the pipeline's
    extracted spans.  Matching is by ``(document_id, label, value)`` —
    exact value match, case-insensitive and stripped.

    Args:
        f1_target: Minimum F1 score for ``NERKPIReport.meets_target``.
            Defaults to 0.85 (Month 2 NER target); set 0.94 for Month 3.
    """

    def __init__(self, f1_target: float = 0.85) -> None:
        """Initialize the NER KPI service.

        Args:
            f1_target: F1 threshold to check (``meets_target`` flag).
        """
        self.f1_target = f1_target

    def evaluate(
        self,
        gold: Sequence[EntitySpan],
        predicted: Sequence[EntitySpan],
    ) -> NERKPIReport:
        """Evaluate predicted spans against gold-label spans.

        Args:
            gold: Human-verified entity spans.
            predicted: Pipeline-extracted entity spans.

        Returns:
            A ``NERKPIReport`` with overall and per-label metrics.
        """
        gold_set: Set[Tuple[str, str, str]] = {
            (s.document_id, s.label, s.value.strip().lower()) for s in gold
        }
        pred_set: Set[Tuple[str, str, str]] = {
            (s.document_id, s.label, s.value.strip().lower()) for s in predicted
        }

        labels = sorted({s.label for s in list(gold) + list(predicted)})
        per_label: List[NERFieldMetrics] = []

        total_tp = total_fp = total_fn = 0

        for label in labels:
            gold_label = {t for t in gold_set if t[1] == label}
            pred_label = {t for t in pred_set if t[1] == label}

            tp = len(gold_label & pred_label)
            fp = len(pred_label - gold_label)
            fn = len(gold_label - pred_label)

            total_tp += tp
            total_fp += fp
            total_fn += fn

            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall)
                else 0.0
            )
            per_label.append(
                NERFieldMetrics(
                    label=label,
                    true_positives=tp,
                    false_positives=fp,
                    false_negatives=fn,
                    precision=precision,
                    recall=recall,
                    f1=f1,
                )
            )

        overall_precision = (
            total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
        )
        overall_recall = (
            total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
        )
        overall_f1 = (
            2 * overall_precision * overall_recall / (overall_precision + overall_recall)
            if (overall_precision + overall_recall)
            else 0.0
        )

        doc_ids = {s.document_id for s in list(gold) + list(predicted)}
        report = NERKPIReport(
            document_count=len(doc_ids),
            overall_precision=round(overall_precision, 4),
            overall_recall=round(overall_recall, 4),
            overall_f1=round(overall_f1, 4),
            per_label=per_label,
            meets_target=overall_f1 >= self.f1_target,
            f1_target=self.f1_target,
        )
        logger.info(
            "NER KPI evaluation: P=%.3f R=%.3f F1=%.3f meets_target=%s.",
            report.overall_precision,
            report.overall_recall,
            report.overall_f1,
            report.meets_target,
        )
        return report

    def from_field_decisions(
        self,
        document_id: str,
        gold_fields: Dict[str, str],
        pipeline_fields: Sequence[FieldDecision],
    ) -> NERKPIReport:
        """Build a KPI report from a single document's gold labels and pipeline output.

        Args:
            document_id: Source document identifier.
            gold_fields: Mapping of ``{label: value}`` from human annotation.
            pipeline_fields: Field decisions produced by the pipeline.

        Returns:
            A ``NERKPIReport`` for this document.
        """
        gold_spans = [
            EntitySpan(document_id=document_id, label=label, value=value)
            for label, value in gold_fields.items()
        ]
        pred_spans = [
            EntitySpan(document_id=document_id, label=fd.field_name, value=fd.value or "")
            for fd in pipeline_fields
            if fd.value
        ]
        return self.evaluate(gold_spans, pred_spans)


# ──────────────────────────────────────────────────────────────────────────────
# New: Extraction storage KPI (Month 3)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ExtractionStorageKPI:
    """Derived KPIs computed from persisted extraction records.

    Attributes:
        total_documents: Total extraction results stored.
        auto_rate: Fraction of documents fully auto-approved.
        review_rate: Fraction requiring human review.
        reject_rate: Fraction rejected.
        average_confidence_by_field: Per-field average confidence scores.
        scorer_distribution: Counts of ``heuristic`` vs ``rf`` scoring mode.
    """

    total_documents: int
    auto_rate: float
    review_rate: float
    reject_rate: float
    average_confidence_by_field: Dict[str, float]
    scorer_distribution: Dict[str, int]


class ExtractionKPIService:
    """Compute KPIs from counts and averages already fetched from the DB.

    Accepts pre-aggregated data so the service stays database-agnostic
    (the repository methods do the SQL work).

    Usage::

        from app.database import ExtractionResultRepository, get_session_factory
        with get_session_factory(settings)() as session:
            repo = ExtractionResultRepository(session)
            counts = repo.count_by_decision()
            avg_conf = repo.average_confidence_by_field()

        kpi = ExtractionKPIService().from_aggregates(counts, avg_conf)
    """

    def from_aggregates(
        self,
        decision_counts: Dict[str, int],
        avg_confidence_by_field: Optional[Dict[str, float]] = None,
        scorer_distribution: Optional[Dict[str, int]] = None,
    ) -> ExtractionStorageKPI:
        """Build a storage KPI report from pre-aggregated counts.

        Args:
            decision_counts: ``{"auto": n, "review": m, "reject": k}``.
            avg_confidence_by_field: Per-field average confidence from DB.
            scorer_distribution: ``{"heuristic": n, "rf": m}``.

        Returns:
            An ``ExtractionStorageKPI`` summary.
        """
        auto = decision_counts.get("auto", 0)
        review = decision_counts.get("review", 0)
        reject = decision_counts.get("reject", 0)
        total = auto + review + reject

        auto_rate = auto / total if total else 0.0
        review_rate = review / total if total else 0.0
        reject_rate = reject / total if total else 0.0

        report = ExtractionStorageKPI(
            total_documents=total,
            auto_rate=round(auto_rate, 4),
            review_rate=round(review_rate, 4),
            reject_rate=round(reject_rate, 4),
            average_confidence_by_field=avg_confidence_by_field or {},
            scorer_distribution=scorer_distribution or {},
        )
        logger.info(
            "ExtractionStorageKPI: total=%d auto_rate=%.3f review_rate=%.3f.",
            report.total_documents,
            report.auto_rate,
            report.review_rate,
        )
        return report


# ──────────────────────────────────────────────────────────────────────────────
# Original PipelineKPIService (unchanged)
# ──────────────────────────────────────────────────────────────────────────────


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
            for fd in document.result.fields:
                field_totals.setdefault(
                    fd.field_name,
                    {
                        "total": 0.0,
                        "auto": 0.0,
                        "review": 0.0,
                        "reject": 0.0,
                        "confidence_sum": 0.0,
                    },
                )
                stats = field_totals[fd.field_name]
                stats["total"] += 1
                stats[fd.decision] += 1
                stats["confidence_sum"] += fd.confidence

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
            fd.confidence
            for document in batch_result.documents
            for fd in document.result.fields
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
                "field_name": f.field_name,
                "total_occurrences": f.total_occurrences,
                "auto_count": f.auto_count,
                "review_count": f.review_count,
                "reject_count": f.reject_count,
                "average_confidence": f.average_confidence,
            }
            for f in report.field_kpis
        ],
    }
