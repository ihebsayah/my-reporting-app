"""Pydantic schemas for API request and response payloads."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """Standard API error payload."""

    detail: str
    error_code: str


class HealthResponse(BaseModel):
    """Health endpoint response."""

    status: str
    environment: str


class ResponseMetadata(BaseModel):
    """Traceability metadata attached to API responses."""

    processed_at: str
    app_version: str
    pipeline_version: str
    extraction_version: str
    model_version: str


class TextRequest(BaseModel):
    """Single text input request."""

    text: str = Field(..., min_length=1)


class BatchTextRequest(BaseModel):
    """Batch text input request."""

    texts: List[str] = Field(..., min_length=1)
    document_ids: Optional[List[str]] = None


class EntityResponse(BaseModel):
    """Serialized extracted entity."""

    start: int
    end: int
    text: str
    label: str
    sources: List[str]
    score: float


class ExtractionResponse(BaseModel):
    """Extraction endpoint response."""

    text: str
    entities: List[EntityResponse]
    metadata: ResponseMetadata


class FieldDecisionResponse(BaseModel):
    """Serialized field-level pipeline decision."""

    field_name: str
    value: Optional[str]
    confidence: float
    decision: str
    sources: List[str]
    confidence_factors: Dict[str, float]
    start: Optional[int]
    end: Optional[int]


class PipelineResponse(BaseModel):
    """Single-document pipeline response."""

    overall_decision: str
    fields: List[FieldDecisionResponse]
    metadata: ResponseMetadata


class BatchDocumentResponse(BaseModel):
    """Per-document batch pipeline result."""

    document_id: str
    overall_decision: str
    field_count: int


class BatchMetricsResponse(BaseModel):
    """Aggregated batch metrics response."""

    document_count: int
    overall_decisions: Dict[str, int]
    field_decisions: Dict[str, Dict[str, int]]


class BatchPipelineResponse(BaseModel):
    """Batch pipeline response."""

    documents: List[BatchDocumentResponse]
    metrics: BatchMetricsResponse
    metadata: ResponseMetadata


class AsyncBatchSubmitResponse(BaseModel):
    """Async batch submission response."""

    job_id: str
    status: str
    submitted_at: str
    metadata: ResponseMetadata


class AsyncBatchStatusResponse(BaseModel):
    """Async batch status response."""

    job_id: str
    status: str
    submitted_at: str
    completed_at: Optional[str]
    error_message: Optional[str]
    result: Optional[BatchPipelineResponse] = None


class FieldKPIResponse(BaseModel):
    """Serialized field KPI metrics."""

    field_name: str
    total_occurrences: int
    auto_count: int
    review_count: int
    reject_count: int
    average_confidence: float


class KPIResponse(BaseModel):
    """KPI report response."""

    document_count: int
    auto_documents: int
    review_documents: int
    reject_documents: int
    average_field_confidence: float
    field_kpis: List[FieldKPIResponse]
    metadata: ResponseMetadata


class AdminStatusResponse(BaseModel):
    """Admin system status response."""

    app_name: str
    app_version: str
    environment: str
    debug: bool
    ner_model_path: str
    ner_model_exists: bool
    pipeline_version: str
    extraction_version: str
    model_version: str
    default_thresholds: Dict[str, float]
    field_thresholds: Dict[str, Dict[str, float]]


class AdminMetricsResponse(BaseModel):
    """Admin metrics summary response."""

    document_count: int
    auto_documents: int
    review_documents: int
    reject_documents: int
    average_field_confidence: float
    field_kpis: List[FieldKPIResponse]
    metadata: ResponseMetadata


class AdminModelResponse(BaseModel):
    """Admin model inspection response."""

    ner_model_path: str
    ner_model_exists: bool
    spacy_available: bool
    train_iterations: int
    model_version: str
