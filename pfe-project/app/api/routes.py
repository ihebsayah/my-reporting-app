"""FastAPI routes for extraction, pipeline, KPI, and admin services."""

import importlib.util
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.api.schemas import (
    AdminModelResponse,
    AdminMetricsResponse,
    AdminStatusResponse,
    AsyncBatchStatusResponse,
    AsyncBatchSubmitResponse,
    BatchDocumentResponse,
    BatchMetricsResponse,
    BatchPipelineResponse,
    BatchTextRequest,
    EntityResponse,
    ErrorResponse,
    ExtractionResponse,
    FieldDecisionResponse,
    FieldKPIResponse,
    KPIResponse,
    PipelineResponse,
    ResponseMetadata,
    TextRequest,
)
from app.config import get_settings
from app.database import AsyncBatchJobRepository, get_session_factory, init_database
from app.kpi import PipelineKPIService, kpi_report_to_payload
from app.ml.ner_extractor import RegexSpacyEnsembleExtractor
from app.pipeline.batch_processor import BatchProcessingResult
from app.pipeline.batch_processor import PipelineBatchProcessor
from app.pipeline.decision_engine import SequentialExtractionDecisionEngine

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/v1")
extractor = RegexSpacyEnsembleExtractor(settings=settings)
pipeline_engine = SequentialExtractionDecisionEngine(settings=settings, extractor=extractor)
batch_processor = PipelineBatchProcessor(settings=settings, engine=pipeline_engine)
kpi_service = PipelineKPIService()
session_factory = get_session_factory(settings)
init_database(settings)

ERROR_RESPONSES = {
    400: {"model": ErrorResponse, "description": "Invalid request payload."},
    422: {"model": ErrorResponse, "description": "Validation error."},
    500: {"model": ErrorResponse, "description": "Internal server error."},
}


@router.post(
    "/extract",
    response_model=ExtractionResponse,
    responses=ERROR_RESPONSES,
    tags=["extraction"],
)
def extract_entities(request: TextRequest) -> ExtractionResponse:
    """Extract entities from raw text."""
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


@router.post(
    "/pipeline/run",
    response_model=PipelineResponse,
    responses=ERROR_RESPONSES,
    tags=["pipeline"],
)
def run_pipeline(request: TextRequest) -> PipelineResponse:
    """Run the sequential extraction pipeline on one document."""
    logger.info("Received single-document pipeline request.")
    result = pipeline_engine.run(request.text)
    return PipelineResponse(
        overall_decision=result.overall_decision,
        fields=[
            FieldDecisionResponse(
                field_name=field.field_name,
                value=field.value,
                confidence=field.confidence,
                decision=field.decision,
                sources=field.sources,
                confidence_factors=field.confidence_factors,
                start=field.start,
                end=field.end,
            )
            for field in result.fields
        ],
        metadata=_build_response_metadata(),
    )


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


@router.post(
    "/kpi/report",
    response_model=KPIResponse,
    responses=ERROR_RESPONSES,
    tags=["kpi"],
)
def build_kpi_report(request: BatchTextRequest) -> KPIResponse:
    """Build a KPI report from a batch of texts."""
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
    return AdminModelResponse(
        ner_model_path=str(model_path),
        ner_model_exists=model_path.exists(),
        spacy_available=importlib.util.find_spec("spacy") is not None,
        train_iterations=settings.ner_train_iterations,
        model_version=settings.model_version,
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
        raise HTTPException(status_code=400, detail="Default source document directory was not found.")
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


def _process_batch_job(
    job_id: str,
    texts: List[str],
    document_ids: Optional[List[str]],
) -> None:
    """Execute an async batch pipeline job and persist its result."""
    with session_factory() as session:
        repository = AsyncBatchJobRepository(session)
        repository.mark_running(job_id)
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
