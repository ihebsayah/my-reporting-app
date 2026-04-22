"""FastAPI routes for extraction, pipeline, KPI, admin, and feedback services.

Month 3 additions
-----------------
* Every ``POST /v1/pipeline/run`` result is now persisted to the DB via
  ``ExtractionResultRepository``.
* ``POST /v1/feedback`` — human-in-the-loop correction endpoint.  Records
  are appended to a JSONL file (``artifacts/feedback/feedback.jsonl``) and
  fed into the next monthly RF retraining run.
* ``POST /v1/kpi/ner`` — evaluate predicted spans vs gold labels and return
  precision/recall/F1 via ``NERKPIService``.
* ``GET  /v1/kpi/storage`` — auto/review/reject rates from persisted DB
  records via ``ExtractionKPIService``.
* ``GET  /v1/extractions/{document_id}`` — historical pipeline results for
  one document from the DB.
* ``RFModelLoader`` is invoked once at module load to hot-wire the engine
  with the latest trained RF model when one is available on disk.
"""

import importlib.util
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.api.agent_bridge import agents_enabled as _agents_enabled
from app.api.agent_bridge import call_agent_service
from app.api.schemas import (
    AdminJobListResponse,
    AdminModelResponse,
    AdminMetricsResponse,
    AdminStatusResponse,
    AgentEnrichedPipelineResponse,
    AsyncBatchJobSummaryResponse,
    AsyncBatchStatusResponse,
    AsyncBatchSubmitResponse,
    BatchDocumentResponse,
    BatchMetricsResponse,
    BatchPipelineResponse,
    BatchTextRequest,
    DriftResponse,
    DriftWindowStats,
    EntityResponse,
    ErrorResponse,
    ExtractionFieldHistoryResponse,
    ExtractionHistoryResponse,
    ExtractionResponse,
    ExtractionStorageKPIResponse,
    FeedbackRequest,
    FeedbackResponse,
    FieldDecisionResponse,
    FieldKPIResponse,
    KPIResponse,
    NERKPIFieldResponse,
    NERKPIResponse,
    PipelineResponse,
    ResponseMetadata,
    RetrainingResponse,
    TextRequest,
)
from app.config import get_settings
from app.database import (
    AsyncBatchJobRepository,
    ExtractionResultRepository,
    FieldRecord,
    get_session_factory,
    init_database,
)
from app.kpi import (
    ExtractionKPIService,
    NERKPIService,
    PipelineKPIService,
    kpi_report_to_payload,
)
from app.kpi.metrics import EntitySpan
from app.ml.ner_extractor import RegexSpacyEnsembleExtractor
from app.pipeline.batch_processor import BatchProcessingResult, PipelineBatchProcessor
from app.pipeline.decision_engine import RFModelLoader, SequentialExtractionDecisionEngine

logger = logging.getLogger(__name__)
settings = get_settings()

# ──────────────────────────────────────────────────────────────────────────────
# Singletons (modules-level, re-used across requests)
# ──────────────────────────────────────────────────────────────────────────────

extractor = RegexSpacyEnsembleExtractor(settings=settings)
pipeline_engine = SequentialExtractionDecisionEngine(settings=settings, extractor=extractor)
batch_processor = PipelineBatchProcessor(settings=settings, engine=pipeline_engine)
kpi_service = PipelineKPIService()
session_factory = get_session_factory(settings)
init_database(settings)

# Attempt to auto-load the latest trained RF model at startup.
_rf_loader = RFModelLoader(settings=settings, use_bert=False)
_startup_rf_model = _rf_loader.load_latest()
if _startup_rf_model is not None:
    pipeline_engine.switch_to_rf_model(_startup_rf_model)
    logger.info("API startup: RF confidence model loaded and active.")
else:
    logger.info("API startup: No RF model found — using heuristic scoring.")

router = APIRouter(prefix="/v1")

ERROR_RESPONSES = {
    400: {"model": ErrorResponse, "description": "Invalid request payload."},
    422: {"model": ErrorResponse, "description": "Validation error."},
    500: {"model": ErrorResponse, "description": "Internal server error."},
}

# Feedback storage path (written to disk, consumed by monthly retraining).
_FEEDBACK_DIR = Path(settings.rf_model_output_dir).parent.parent / "feedback"


# ──────────────────────────────────────────────────────────────────────────────
# Extraction
# ──────────────────────────────────────────────────────────────────────────────


@router.post(
    "/extract",
    response_model=ExtractionResponse,
    responses=ERROR_RESPONSES,
    tags=["extraction"],
)
def extract_entities(request: TextRequest) -> ExtractionResponse:
    """Extract entities from raw text without running the decision pipeline."""
    logger.info("Received extraction request.")
    result = extractor.extract(request.text)
    return ExtractionResponse(
        text=result.text,
        entities=[
            EntityResponse(
                start=entity.start,
                end=entity.end,
                text=entity.text,
                label=entity.label,
                sources=list(entity.sources),
                score=entity.score,
            )
            for entity in result.entities
        ],
        metadata=_build_response_metadata(),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline (single document)
# ──────────────────────────────────────────────────────────────────────────────


@router.post(
    "/pipeline/run",
    response_model=AgentEnrichedPipelineResponse,
    responses=ERROR_RESPONSES,
    tags=["pipeline"],
)
def run_pipeline(request: TextRequest) -> AgentEnrichedPipelineResponse:
    """Run the sequential extraction pipeline on one document and persist the result.

    When ``AGENT_SERVICE_URL`` is set in the environment, this endpoint also
    calls the AI agent service and includes the agent decision + reasoning in
    the ``agent`` field of the response.  If the agent service is unreachable
    or disabled, ``agent=None`` and the response is identical to the original
    pipeline output — the existing behaviour is fully preserved.
    """
    logger.info("Received single-document pipeline request.")
    result = pipeline_engine.run(request.text)

    # Derive a stable document_id for traceability.
    doc_id = request.text[:40].replace(" ", "_").replace("\n", "")

    # Persist every run to the extraction_results table.
    try:
        field_records = [
            FieldRecord(
                field_name=fd.field_name,
                value=fd.value,
                confidence=fd.confidence,
                decision=fd.decision,
                sources=list(fd.sources),
                scorer=fd.scorer,
                char_start=fd.start,
                char_end=fd.end,
            )
            for fd in result.fields
        ]
        with session_factory() as session:
            ExtractionResultRepository(session).save(
                document_id=doc_id,
                source_text=request.text,
                overall_decision=result.overall_decision,
                scorer=result.scorer,
                model_version=settings.model_version,
                fields=field_records,
            )
    except Exception as persist_exc:
        logger.warning("Failed to persist pipeline result: %s.", persist_exc)

    # ── Agent service call (Canary — additive, never blocks) ──────────────
    agent_decision = call_agent_service(
        text=request.text,
        document_id=doc_id,
        main_decision=result.overall_decision,
    )
    if agent_decision:
        logger.info(
            "Agent enrichment: action=%s confidence=%.3f doc_id=%s.",
            agent_decision.action,
            agent_decision.confidence,
            doc_id,
        )

    return AgentEnrichedPipelineResponse(
        overall_decision=result.overall_decision,
        scorer=result.scorer,
        fields=[
            FieldDecisionResponse(
                field_name=fd.field_name,
                value=fd.value,
                confidence=fd.confidence,
                decision=fd.decision,
                sources=fd.sources,
                confidence_factors=fd.confidence_factors,
                start=fd.start,
                end=fd.end,
            )
            for fd in result.fields
        ],
        metadata=_build_response_metadata(),
        agent=agent_decision,
        agent_enabled=_agents_enabled(),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline (batch)
# ──────────────────────────────────────────────────────────────────────────────


@router.post(
    "/pipeline/batch",
    response_model=BatchPipelineResponse,
    responses=ERROR_RESPONSES,
    tags=["pipeline"],
)
def run_pipeline_batch(request: BatchTextRequest) -> BatchPipelineResponse:
    """Run the sequential extraction pipeline on a batch of texts."""
    logger.info("Received batch pipeline request for %d documents.", len(request.texts))
    if request.document_ids is not None and len(request.document_ids) != len(request.texts):
        raise HTTPException(status_code=400, detail="document_ids length must match texts length.")
    result = batch_processor.run_texts(request.texts, request.document_ids)
    return _serialize_batch_result(result)


@router.post(
    "/pipeline/batch/submit",
    response_model=AsyncBatchSubmitResponse,
    responses=ERROR_RESPONSES,
    tags=["pipeline"],
)
def submit_pipeline_batch(
    request: BatchTextRequest,
    background_tasks: BackgroundTasks,
) -> AsyncBatchSubmitResponse:
    """Submit a batch pipeline job for asynchronous processing."""
    logger.info("Received async batch submission for %d documents.", len(request.texts))
    if request.document_ids is not None and len(request.document_ids) != len(request.texts):
        raise HTTPException(status_code=400, detail="document_ids length must match texts length.")

    with session_factory() as session:
        job = AsyncBatchJobRepository(session).create_job()
    background_tasks.add_task(
        _process_batch_job,
        job.job_id,
        request.texts,
        request.document_ids,
    )
    return AsyncBatchSubmitResponse(
        job_id=job.job_id,
        status=job.status,
        submitted_at=job.submitted_at,
        metadata=_build_response_metadata(),
    )


@router.get(
    "/pipeline/batch/jobs/{job_id}",
    response_model=AsyncBatchStatusResponse,
    responses=ERROR_RESPONSES,
    tags=["pipeline"],
)
def get_pipeline_batch_job(job_id: str) -> AsyncBatchStatusResponse:
    """Return the current state of an async batch pipeline job."""
    with session_factory() as session:
        job = AsyncBatchJobRepository(session).get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Batch job was not found.")
    return AsyncBatchStatusResponse(
        job_id=job.job_id,
        status=job.status,
        submitted_at=job.submitted_at,
        completed_at=job.completed_at,
        error_message=job.error_message,
        result=BatchPipelineResponse(**job.result_payload) if job.result_payload else None,
    )


# ──────────────────────────────────────────────────────────────────────────────
# KPI endpoints
# ──────────────────────────────────────────────────────────────────────────────


@router.post(
    "/kpi/report",
    response_model=KPIResponse,
    responses=ERROR_RESPONSES,
    tags=["kpi"],
)
def build_kpi_report(request: BatchTextRequest) -> KPIResponse:
    """Build a pipeline KPI report from a batch of texts."""
    logger.info("Received KPI report request for %d documents.", len(request.texts))
    if request.document_ids is not None and len(request.document_ids) != len(request.texts):
        raise HTTPException(status_code=400, detail="document_ids length must match texts length.")
    batch_result = batch_processor.run_texts(request.texts, request.document_ids)
    report = kpi_service.build_report(batch_result)
    payload = kpi_report_to_payload(report)
    return KPIResponse(
        document_count=payload["document_count"],
        auto_documents=payload["auto_documents"],
        review_documents=payload["review_documents"],
        reject_documents=payload["reject_documents"],
        average_field_confidence=payload["average_field_confidence"],
        field_kpis=[FieldKPIResponse(**item) for item in payload["field_kpis"]],
        metadata=_build_response_metadata(),
    )


@router.post(
    "/kpi/ner",
    response_model=NERKPIResponse,
    responses=ERROR_RESPONSES,
    tags=["kpi"],
)
def build_ner_kpi_report(
    gold: List[dict],
    predicted: List[dict],
    f1_target: float = 0.85,
) -> NERKPIResponse:
    """Evaluate NER extraction quality against gold-label spans.

    Each item in ``gold`` and ``predicted`` must have:
    ``document_id``, ``label``, ``value`` keys.

    Args:
        gold: Gold-label entity spans.
        predicted: Pipeline-predicted entity spans.
        f1_target: F1 threshold for ``meets_target`` flag (default 0.85).
    """
    logger.info(
        "Received NER KPI request: %d gold, %d predicted spans.",
        len(gold),
        len(predicted),
    )
    try:
        gold_spans = [EntitySpan(**item) for item in gold]
        pred_spans = [EntitySpan(**item) for item in predicted]
    except (TypeError, KeyError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Each span must have document_id, label, value keys. Error: {exc}",
        ) from exc

    report = NERKPIService(f1_target=f1_target).evaluate(gold_spans, pred_spans)
    return NERKPIResponse(
        document_count=report.document_count,
        overall_precision=report.overall_precision,
        overall_recall=report.overall_recall,
        overall_f1=report.overall_f1,
        meets_target=report.meets_target,
        f1_target=report.f1_target,
        per_label=[
            NERKPIFieldResponse(
                label=m.label,
                true_positives=m.true_positives,
                false_positives=m.false_positives,
                false_negatives=m.false_negatives,
                precision=m.precision,
                recall=m.recall,
                f1=m.f1,
            )
            for m in report.per_label
        ],
        metadata=_build_response_metadata(),
    )


@router.get(
    "/kpi/storage",
    response_model=ExtractionStorageKPIResponse,
    responses=ERROR_RESPONSES,
    tags=["kpi"],
)
def build_storage_kpi() -> ExtractionStorageKPIResponse:
    """Return auto/review/reject rates and per-field confidence from persisted DB records."""
    logger.info("Received storage KPI request.")
    with session_factory() as session:
        repo = ExtractionResultRepository(session)
        decision_counts = repo.count_by_decision()
        avg_conf = repo.average_confidence_by_field()

    report = ExtractionKPIService().from_aggregates(
        decision_counts=decision_counts,
        avg_confidence_by_field=avg_conf,
    )
    return ExtractionStorageKPIResponse(
        total_documents=report.total_documents,
        auto_rate=report.auto_rate,
        review_rate=report.review_rate,
        reject_rate=report.reject_rate,
        average_confidence_by_field=report.average_confidence_by_field,
        scorer_distribution=report.scorer_distribution,
        metadata=_build_response_metadata(),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Extraction history
# ──────────────────────────────────────────────────────────────────────────────


@router.get(
    "/extractions/{document_id}",
    response_model=List[ExtractionHistoryResponse],
    responses=ERROR_RESPONSES,
    tags=["extraction"],
)
def get_extraction_history(document_id: str, limit: int = 20) -> List[ExtractionHistoryResponse]:
    """Return historical pipeline results for a document from the DB."""
    logger.info("Received extraction history request for document_id=%s.", document_id)
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 200.")
    with session_factory() as session:
        records = ExtractionResultRepository(session).list_by_document(document_id, limit=limit)
    return [
        ExtractionHistoryResponse(
            record_id=record.id,
            document_id=record.document_id,
            overall_decision=record.overall_decision,
            scorer=record.scorer,
            model_version=record.model_version,
            processed_at=record.processed_at,
            fields=[
                ExtractionFieldHistoryResponse(
                    field_name=f.field_name,
                    value=f.value,
                    confidence=f.confidence,
                    decision=f.decision,
                    sources=f.sources,
                    scorer=f.scorer,
                )
                for f in record.fields
            ],
            metadata=_build_response_metadata(),
        )
        for record in records
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Feedback (human-in-the-loop)
# ──────────────────────────────────────────────────────────────────────────────


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    responses=ERROR_RESPONSES,
    tags=["feedback"],
)
def submit_feedback(request: FeedbackRequest) -> FeedbackResponse:
    """Record a human correction for a field extraction.

    Corrections are appended to ``artifacts/feedback/feedback.jsonl`` and
    consumed during the next monthly RF retraining run (Month 4).
    """
    logger.info(
        "Received feedback for document_id=%s field=%s.",
        request.document_id,
        request.field_name,
    )
    try:
        _append_feedback(request)
        recorded = True
        message = "Feedback recorded successfully."
    except Exception as exc:
        logger.error("Failed to persist feedback: %s.", exc)
        recorded = False
        message = f"Feedback could not be persisted: {exc}"

    return FeedbackResponse(
        document_id=request.document_id,
        field_name=request.field_name,
        correct_value=request.correct_value,
        recorded=recorded,
        message=message,
        metadata=_build_response_metadata(),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Monitoring
# ──────────────────────────────────────────────────────────────────────────────


@router.get(
    "/monitoring/drift",
    response_model=DriftResponse,
    responses=ERROR_RESPONSES,
    tags=["monitoring"],
)
def get_drift_report(
    baseline_window: int = 50,
    recent_window: int = 20,
    auto_threshold: float = 0.10,
    confidence_threshold: float = 0.05,
) -> DriftResponse:
    """Run a sliding-window confidence and auto-rate drift check from DB records.

    Returns ``insufficient_history=True`` when fewer than
    ``baseline_window + recent_window`` extraction records are stored.
    """
    from app.monitoring.drift import ConfidenceDriftDetector

    needed = baseline_window + recent_window
    with session_factory() as session:
        repo = ExtractionResultRepository(session)
        records = (
            repo.list_by_decision("auto", limit=needed)
            + repo.list_by_decision("review", limit=needed)
            + repo.list_by_decision("reject", limit=needed)
        )

    detector = ConfidenceDriftDetector(
        auto_rate_drop_threshold=auto_threshold,
        confidence_drop_threshold=confidence_threshold,
        baseline_window_size=baseline_window,
        recent_window_size=recent_window,
    )
    samples = detector.samples_from_records(records)
    report = detector.detect(samples)
    insufficient = len(samples) < needed

    return DriftResponse(
        drift_detected=report.drift_detected,
        triggered_signals=report.triggered_signals,
        auto_rate_drop=report.auto_rate_drop,
        confidence_drop=report.confidence_drop,
        auto_rate_threshold=report.auto_rate_threshold,
        confidence_threshold=report.confidence_threshold,
        baseline=DriftWindowStats(
            window_size=report.baseline.window_size,
            auto_rate=report.baseline.auto_rate,
            review_rate=report.baseline.review_rate,
            reject_rate=report.baseline.reject_rate,
            mean_confidence=report.baseline.mean_confidence,
        ) if not insufficient else None,
        recent=DriftWindowStats(
            window_size=report.recent.window_size,
            auto_rate=report.recent.auto_rate,
            review_rate=report.recent.review_rate,
            reject_rate=report.recent.reject_rate,
            mean_confidence=report.recent.mean_confidence,
        ) if not insufficient else None,
        checked_at=report.checked_at,
        insufficient_history=insufficient,
        metadata=_build_response_metadata(),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Admin endpoints
# ──────────────────────────────────────────────────────────────────────────────


@router.get(
    "/admin/status",
    response_model=AdminStatusResponse,
    responses={500: ERROR_RESPONSES[500]},
    tags=["admin"],
)
def admin_status() -> AdminStatusResponse:
    """Return application and model configuration status."""
    model_path = Path(settings.ner_model_path)
    return AdminStatusResponse(
        app_name=settings.app_name,
        app_version=settings.app_version,
        environment=settings.app_env,
        debug=settings.app_debug,
        ner_model_path=str(model_path),
        ner_model_exists=model_path.exists(),
        pipeline_version=settings.pipeline_version,
        extraction_version=settings.extraction_version,
        model_version=settings.model_version,
        default_thresholds={
            "auto": settings.auto_approval_threshold,
            "review_min": settings.review_min_threshold,
            "review_max": settings.review_max_threshold,
        },
        field_thresholds=settings.field_thresholds(),
    )


@router.get(
    "/admin/model",
    response_model=AdminModelResponse,
    responses={500: ERROR_RESPONSES[500]},
    tags=["admin"],
)
def admin_model() -> AdminModelResponse:
    """Return model availability and training configuration details."""
    model_path = Path(settings.ner_model_path)
    available_rf = _rf_loader.list_available()
    return AdminModelResponse(
        ner_model_path=str(model_path),
        ner_model_exists=model_path.exists(),
        spacy_available=importlib.util.find_spec("spacy") is not None,
        train_iterations=settings.ner_train_iterations,
        model_version=settings.model_version,
        rf_model_available=len(available_rf) > 0,
        rf_model_version=available_rf[0] if available_rf else None,
    )


@router.post(
    "/admin/retrain",
    response_model=RetrainingResponse,
    responses=ERROR_RESPONSES,
    tags=["admin"],
)
def admin_trigger_retrain(n_estimators: int = 200) -> RetrainingResponse:
    """Trigger the monthly RF retraining pipeline from the API.

    Reads ``docs/annotation/rf_training_records.jsonl`` as the base dataset,
    merges any human feedback from ``artifacts/feedback/feedback.jsonl``,
    trains a new RF model, and hot-swaps the live pipeline engine.

    Args:
        n_estimators: Number of trees in the new RF (default 200).
    """
    from app.retraining.pipeline import RFRetrainingPipeline

    logger.info("Admin retrain endpoint triggered (n_estimators=%d).", n_estimators)
    try:
        retrain_pipeline = RFRetrainingPipeline(settings=settings, use_bert=False)
        result = retrain_pipeline.run(
            base_jsonl_path=Path("docs/annotation/rf_training_records.jsonl"),
            output_dir=Path(settings.rf_model_output_dir),
            n_estimators=n_estimators,
            engine=pipeline_engine,
        )
        return RetrainingResponse(
            success=True,
            model_version=result.training_result.model_version,
            accuracy=result.training_result.accuracy,
            total_records=result.total_records,
            feedback_records=result.feedback_records,
            model_path=result.model_path,
            metadata=_build_response_metadata(),
        )
    except Exception as exc:
        logger.error("Admin retrain failed: %s.", exc)
        return RetrainingResponse(
            success=False,
            error=str(exc),
            metadata=_build_response_metadata(),
        )


@router.get(
    "/admin/metrics",
    response_model=AdminMetricsResponse,
    responses=ERROR_RESPONSES,
    tags=["admin"],
)
def admin_metrics() -> AdminMetricsResponse:
    """Build KPI metrics from the default source-document directory."""
    source_dir = "docs/source_documents"
    if not Path(source_dir).exists():
        raise HTTPException(
            status_code=400,
            detail="Default source document directory was not found.",
        )
    batch_result = batch_processor.run_directory(source_dir)
    report = kpi_service.build_report(batch_result)
    payload = kpi_report_to_payload(report)
    return AdminMetricsResponse(
        document_count=payload["document_count"],
        auto_documents=payload["auto_documents"],
        review_documents=payload["review_documents"],
        reject_documents=payload["reject_documents"],
        average_field_confidence=payload["average_field_confidence"],
        field_kpis=[FieldKPIResponse(**item) for item in payload["field_kpis"]],
        metadata=_build_response_metadata(),
    )


@router.get(
    "/admin/jobs",
    response_model=AdminJobListResponse,
    responses=ERROR_RESPONSES,
    tags=["admin"],
)
def admin_list_jobs(
    status: Optional[str] = None,
    limit: int = 20,
) -> AdminJobListResponse:
    """List recent async batch jobs for operators."""
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 100.")
    with session_factory() as session:
        jobs = AsyncBatchJobRepository(session).list_jobs(status=status, limit=limit)
    return AdminJobListResponse(
        jobs=[
            AsyncBatchJobSummaryResponse(
                job_id=job.job_id,
                status=job.status,
                submitted_at=job.submitted_at,
                completed_at=job.completed_at,
                error_message=job.error_message,
            )
            for job in jobs
        ],
        metadata=_build_response_metadata(),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────


def _process_batch_job(
    job_id: str,
    texts: List[str],
    document_ids: Optional[List[str]],
) -> None:
    """Execute an async batch pipeline job and persist its result."""
    with session_factory() as session:
        AsyncBatchJobRepository(session).mark_running(job_id)
    try:
        result = batch_processor.run_texts(texts, document_ids)
    except Exception as exc:
        logger.exception("Async batch pipeline job %s failed.", job_id, exc_info=exc)
        with session_factory() as session:
            AsyncBatchJobRepository(session).mark_failed(job_id, str(exc))
        return
    payload = _serialize_batch_result(result).model_dump()
    with session_factory() as session:
        AsyncBatchJobRepository(session).mark_completed(job_id, payload)


def _serialize_batch_result(
    result: Optional[BatchProcessingResult],
) -> Optional[BatchPipelineResponse]:
    """Serialize a batch processing result into the API response shape."""
    if result is None:
        return None
    return BatchPipelineResponse(
        documents=[
            BatchDocumentResponse(
                document_id=document.document_id,
                overall_decision=document.result.overall_decision,
                field_count=len(document.result.fields),
            )
            for document in result.documents
        ],
        metrics=BatchMetricsResponse(
            document_count=result.metrics.document_count if result.metrics else 0,
            overall_decisions=result.metrics.overall_decisions if result.metrics else {},
            field_decisions=result.metrics.field_decisions if result.metrics else {},
        ),
        metadata=_build_response_metadata(),
    )


def _build_response_metadata() -> ResponseMetadata:
    """Build a shared response metadata payload for API traceability."""
    return ResponseMetadata(
        processed_at=datetime.now(timezone.utc).isoformat(),
        app_version=settings.app_version,
        pipeline_version=settings.pipeline_version,
        extraction_version=settings.extraction_version,
        model_version=settings.model_version,
    )


def _append_feedback(request: FeedbackRequest) -> None:
    """Serialize and append one feedback record to the JSONL feedback file."""
    _FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    feedback_file = _FEEDBACK_DIR / "feedback.jsonl"
    record = {
        "document_id": request.document_id,
        "field_name": request.field_name,
        "correct_value": request.correct_value,
        "original_value": request.original_value,
        "original_decision": request.original_decision,
        "notes": request.notes,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    with feedback_file.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=True) + "\n")
    logger.info("Appended feedback for document_id=%s field=%s.", request.document_id, request.field_name)
