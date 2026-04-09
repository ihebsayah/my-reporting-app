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
    scorer: str = "heuristic"
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


class AsyncBatchJobSummaryResponse(BaseModel):
    """Compact async batch job summary for admin lists."""

    job_id: str
    status: str
    submitted_at: str
    completed_at: Optional[str]
    error_message: Optional[str]


class AdminJobListResponse(BaseModel):
    """Admin async job listing response."""

    jobs: List[AsyncBatchJobSummaryResponse]
    metadata: ResponseMetadata


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
    rf_model_available: bool = False
    rf_model_version: Optional[str] = None


# ── Month 3: feedback, NER KPI, extraction history ────────────────────────────


class FeedbackRequest(BaseModel):
    """Human correction feedback for a single extracted field."""

    document_id: str = Field(..., min_length=1)
    field_name: str = Field(..., min_length=1)
    correct_value: str = Field(..., description="The human-verified correct value for this field.")
    original_value: Optional[str] = Field(None, description="The value the pipeline extracted.")
    original_decision: Optional[str] = Field(None, description="auto / review / reject")
    notes: Optional[str] = None


class FeedbackResponse(BaseModel):
    """Confirmation response after recording a feedback correction."""

    document_id: str
    field_name: str
    correct_value: str
    recorded: bool
    message: str
    metadata: ResponseMetadata


class NERKPIFieldResponse(BaseModel):
    """Per-label NER precision/recall/F1."""

    label: str
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float


class NERKPIResponse(BaseModel):
    """NER KPI report response."""

    document_count: int
    overall_precision: float
    overall_recall: float
    overall_f1: float
    meets_target: bool
    f1_target: float
    per_label: List[NERKPIFieldResponse]
    metadata: ResponseMetadata


class ExtractionFieldHistoryResponse(BaseModel):
    """One extracted field from a historical pipeline result."""

    field_name: str
    value: Optional[str]
    confidence: float
    decision: str
    sources: List[str]
    scorer: str


class ExtractionHistoryResponse(BaseModel):
    """Historical pipeline result for one document."""

    record_id: int
    document_id: str
    overall_decision: str
    scorer: str
    model_version: str
    processed_at: str
    fields: List[ExtractionFieldHistoryResponse]
    metadata: ResponseMetadata


class ExtractionStorageKPIResponse(BaseModel):
    """Extraction storage KPI from persisted DB records."""

    total_documents: int
    auto_rate: float
    review_rate: float
    reject_rate: float
    average_confidence_by_field: Dict[str, float]
    scorer_distribution: Dict[str, int]
    metadata: ResponseMetadata


class DriftWindowStats(BaseModel):
    """Stats for one drift-detection sliding window."""

    window_size: int
    auto_rate: float
    review_rate: float
    reject_rate: float
    mean_confidence: float


class DriftResponse(BaseModel):
    """Drift detection report response."""

    drift_detected: bool
    triggered_signals: List[str]
    auto_rate_drop: float
    confidence_drop: float
    auto_rate_threshold: float
    confidence_threshold: float
    baseline: Optional[DriftWindowStats] = None
    recent: Optional[DriftWindowStats] = None
    checked_at: str
    insufficient_history: bool = False
    metadata: ResponseMetadata


class RetrainingResponse(BaseModel):
    """Admin RF retraining trigger response."""

    success: bool
    model_version: Optional[str] = None
    accuracy: Optional[float] = None
    total_records: Optional[int] = None
    feedback_records: Optional[int] = None
    model_path: Optional[str] = None
    error: Optional[str] = None
    metadata: ResponseMetadata
