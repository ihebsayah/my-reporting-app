"""ML tools — thin wrappers that call the existing FastAPI extraction pipeline.

Agents treat these as LangChain-compatible tool functions.  All computation
stays inside the existing app; agents only interpret the results.
"""

import json
import logging
from typing import Any, Dict, List, Optional

import httpx
from langchain_core.tools import tool

from agents.config import get_agent_settings

logger = logging.getLogger(__name__)
_settings = get_agent_settings()


def _pipeline_post(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """POST to the existing FastAPI pipeline and return the JSON response.

    Args:
        endpoint: Path relative to the API base URL (e.g. ``/api/v1/extract``).
        payload: JSON-serialisable request body.

    Returns:
        Parsed JSON response as a dictionary.

    Raises:
        RuntimeError: If the upstream call fails or returns a non-2xx status.
    """
    url = f"{_settings.api_base_url}{endpoint}"
    try:
        response = httpx.post(url, json=payload, timeout=30.0)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        logger.error("ML tool HTTP error calling %s: %s", url, exc)
        raise RuntimeError(f"Pipeline call failed [{exc.response.status_code}]: {exc}") from exc
    except httpx.RequestError as exc:
        logger.error("ML tool connection error calling %s: %s", url, exc)
        raise RuntimeError(f"Pipeline unreachable: {exc}") from exc


@tool
def run_ner_extraction(text: str) -> str:
    """Run the spaCy + regex NER ensemble extractor on a document text.

    Calls the existing ``POST /api/v1/extract`` endpoint and returns
    extracted entities with their confidence scores.

    Args:
        text: Raw document text to extract entities from.

    Returns:
        JSON string with a list of extracted entities.
    """
    logger.info("run_ner_extraction called (text length=%d).", len(text))
    try:
        result = _pipeline_post("/api/v1/extract", {"text": text})
        entities = result.get("entities", [])
        logger.info("NER extraction returned %d entities.", len(entities))
        return json.dumps({"entities": entities, "entity_count": len(entities)})
    except RuntimeError as exc:
        logger.warning("NER extraction tool failed, returning empty result: %s", exc)
        return json.dumps({"entities": [], "entity_count": 0, "error": str(exc)})


@tool
def run_confidence_scoring(text: str) -> str:
    """Run the full Sequential Extraction Decision pipeline (spaCy + RF confidence).

    Calls ``POST /api/v1/pipeline/run`` which wraps both NER extraction and
    Random Forest confidence scoring. Returns per-field decisions and the
    overall document routing decision.

    Args:
        text: Raw document text.

    Returns:
        JSON string with overall_decision, scorer, and per-field details.
    """
    logger.info("run_confidence_scoring called (text length=%d).", len(text))
    try:
        result = _pipeline_post("/api/v1/pipeline/run", {"text": text})
        output = {
            "overall_decision": result.get("overall_decision"),
            "scorer": result.get("scorer"),
            "fields": result.get("fields", []),
            "field_count": len(result.get("fields", [])),
        }
        logger.info(
            "Confidence scoring: decision=%s, fields=%d",
            output["overall_decision"],
            output["field_count"],
        )
        return json.dumps(output)
    except RuntimeError as exc:
        logger.warning("Confidence scoring tool failed: %s", exc)
        return json.dumps({"overall_decision": "review", "scorer": "error", "fields": [], "error": str(exc)})


@tool
def run_bart_classification(text: str) -> str:
    """Classify the document type using the existing BART zero-shot classifier.

    In the current system the pipeline decision engine infers document type
    from the field composition; this tool runs the pipeline and interprets the
    field mix to guess document type (invoice, receipt, contract, report).

    Args:
        text: Raw document text.

    Returns:
        JSON string with doc_type and confidence estimate.
    """
    logger.info("run_bart_classification called (text length=%d).", len(text))
    try:
        result = _pipeline_post("/api/v1/pipeline/run", {"text": text})
        fields = result.get("fields", [])
        field_names = [f.get("field_name", "") for f in fields]

        # Heuristic document-type inference from extracted field labels.
        doc_type, confidence = _infer_doc_type(field_names, text)
        logger.info("BART classification: doc_type=%s, confidence=%.2f", doc_type, confidence)
        return json.dumps({"doc_type": doc_type, "confidence": confidence, "field_names": field_names})
    except RuntimeError as exc:
        logger.warning("BART classification tool failed: %s", exc)
        return json.dumps({"doc_type": "unknown", "confidence": 0.0, "error": str(exc)})


def _infer_doc_type(field_names: List[str], text: str) -> tuple:
    """Lightweight heuristic document-type inference.

    Args:
        field_names: List of NER field labels from the extraction pipeline.
        text: Raw document text (used for keyword signals).

    Returns:
        Tuple of (doc_type string, confidence float).
    """
    text_lower = text.lower()
    has_invoice_id = "INVOICE_ID" in field_names
    has_amount = "TOTAL_AMOUNT" in field_names
    has_vendor = "VENDOR_NAME" in field_names
    has_date = "INVOICE_DATE" in field_names

    invoice_keywords = ["invoice", "inv #", "bill to", "due date", "subtotal", "tax"]
    receipt_keywords = ["receipt", "thank you for your purchase", "change due"]
    contract_keywords = ["agreement", "clause", "party", "hereby", "obligations"]
    report_keywords = ["report", "summary", "period", "quarterly", "annual", "kpi"]

    def _keyword_score(keywords: List[str]) -> float:
        hits = sum(1 for kw in keywords if kw in text_lower)
        return hits / len(keywords)

    scores = {
        "invoice": _keyword_score(invoice_keywords) + (0.3 if has_invoice_id and has_amount else 0),
        "receipt": _keyword_score(receipt_keywords) + (0.2 if has_amount and not has_vendor else 0),
        "contract": _keyword_score(contract_keywords),
        "report": _keyword_score(report_keywords),
    }

    # Fallback: known field mix suggests invoice.
    if has_invoice_id and has_amount and has_vendor and has_date:
        scores["invoice"] = max(scores["invoice"], 0.85)

    best_type = max(scores, key=lambda k: scores[k])
    best_score = scores[best_type]

    if best_score < 0.1:
        return "unknown", 0.4
    # Normalise to [0.5, 0.99].
    confidence = min(0.99, 0.5 + best_score * 0.49)
    return best_type, round(confidence, 4)
