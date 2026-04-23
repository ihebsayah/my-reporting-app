"""LLM Reasoner — wraps Ollama/Mistral to provide true LLM reasoning inside agents.

This module is the single LLM integration point for the entire agent service.
All four sub-agents call ``LLMReasoner.think()`` to generate structured reasoning.

Design principles:
- **Graceful fallback**: if Ollama is unreachable or times out, heuristic
  reasoning is returned instead — agents always produce output.
- **Structured output**: prompts force JSON or labelled-section responses that
  can be parsed reliably without hallucination risk.
- **Single instance**: the module provides a ``get_reasoner()`` singleton so
  the model connection is shared across all agents in one process.
- **Temperature=0.1**: near-deterministic for consistent business decisions.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from functools import lru_cache
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

_LLM_BASE_URL: str = os.environ.get("LLM_BASE_URL", "http://localhost:11434")
_LLM_MODEL: str    = os.environ.get("LLM_MODEL_NAME", "mistral")
_LLM_TEMPERATURE: float = float(os.environ.get("LLM_TEMPERATURE", "0.1"))
_LLM_MAX_TOKENS: int    = int(os.environ.get("LLM_MAX_TOKENS", "1024"))
_LLM_TIMEOUT: float     = float(os.environ.get("LLM_TIMEOUT", "30.0"))
_LLM_ENABLED: bool      = os.environ.get("LLM_ENABLED", "true").lower() == "true"
_LLM_PROBE_TTL: float   = float(os.environ.get("LLM_PROBE_TTL", "60"))  # re-probe interval (s)


# ── ReAct prompt templates ────────────────────────────────────────────────────

_CLASSIFIER_PROMPT = """\
You are a document classification agent for an invoice processing system.
Given document text and a list of extracted NER field labels, decide what type of document this is.

Document text (first 400 chars):
{text}

Extracted NER labels: {field_names}
BART base classification: {bart_doc_type} (confidence: {bart_confidence:.0%})

Reason step by step, then output a JSON object on the LAST line.
JSON format: {{"doc_type": "invoice|receipt|contract|report|unknown", "confidence": 0.0-1.0, "reasoning": "brief reason"}}

Think:"""

_EXTRACTOR_PROMPT = """\
You are a field extraction agent. A document has been classified as: {doc_type}
The pipeline extracted these fields. Review them and decide which are reliable.

Extracted fields:
{fields_json}

Missing expected fields for a {doc_type}: {missing_fields}

For each extracted field, confirm if it is correct, needs review, or should be rejected.
Then output a JSON object on the LAST line:
{{"overall_decision": "auto|review", "reliable_fields": ["FIELD1", ...], "uncertain_fields": ["FIELD2", ...], "reasoning": "brief"}}

Think:"""

_VALIDATOR_PROMPT = """\
You are a validation agent reviewing extracted invoice data for anomalies.

Vendor: {vendor}
Amount: {amount}
Date: {date}
Vendor history: {vendor_history}
Validation issues found: {issues}

Assess the risk level and whether human review is warranted.
Output a JSON object on the LAST line:
{{"risk_level": "low|medium|high", "confidence_adjustment": -0.1 to +0.05, "human_review_recommended": true|false, "reasoning": "brief"}}

Think:"""

_ROUTER_PROMPT = """\
You are a routing agent making the final document processing decision.

Document type: {doc_type}
Extraction confidence: {extraction_confidence:.0%}
Validation passed: {is_valid}
Vendor known: {vendor_known}
Amount: {amount}
Safety rails triggered: {safety_rails}
Validator risk level: {risk_level}
Auto-approve threshold: {auto_threshold:.0%}
Human-review threshold: {review_threshold:.0%}

Based on all evidence, decide: auto_approve, human_review, or reject.
Safety rails OVERRIDE everything — if any are triggered, you MUST follow them.

Output a JSON object on the LAST line:
{{"action": "auto_approve|human_review|reject", "confidence": 0.0-1.0, "reasoning": "clear explanation for the human operator"}}

Think:"""


# ── LLMReasoner ───────────────────────────────────────────────────────────────


class LLMReasoner:
    """Thin wrapper around OllamaLLM for ReAct-style agent reasoning.

    Args:
        model: Ollama model name (default: ``mistral``).
        base_url: Ollama server URL (default: ``http://localhost:11434``).
        temperature: Sampling temperature (default: 0.1 for near-determinism).
        max_tokens: Maximum response tokens.
        timeout: HTTP timeout for the Ollama call.
    """

    def __init__(
        self,
        model: str = _LLM_MODEL,
        base_url: str = _LLM_BASE_URL,
        temperature: float = _LLM_TEMPERATURE,
        max_tokens: int = _LLM_MAX_TOKENS,
        timeout: float = _LLM_TIMEOUT,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._llm = None
        self._available: Optional[bool] = None  # None = not yet tested
        self._last_probe_at: float = 0.0          # epoch time of last probe

    def _get_llm(self):
        """Lazy-initialise the LangChain OllamaLLM client."""
        if self._llm is not None:
            return self._llm
        try:
            from langchain_ollama import OllamaLLM

            self._llm = OllamaLLM(
                model=self.model,
                base_url=self.base_url,
                temperature=self.temperature,
                num_predict=self.max_tokens,
            )
            logger.info(
                "LLMReasoner: initialised OllamaLLM model=%s base_url=%s.",
                self.model, self.base_url,
            )
            return self._llm
        except ImportError:
            logger.warning(
                "langchain_ollama not installed — LLM reasoning disabled. "
                "Install with: pip install langchain-ollama"
            )
            return None
        except Exception as exc:
            logger.warning("LLMReasoner: failed to initialise OllamaLLM: %s", exc)
            return None

    def is_available(self) -> bool:
        """Check if the Ollama server is reachable.

        Result is cached for ``LLM_PROBE_TTL`` seconds (default 60s) so the
        agent service automatically detects when Ollama comes online without
        requiring a restart.  When Ollama is unavailable agents fall back to
        heuristic reasoning gracefully.

        Returns:
            True if Ollama responds to a health probe within 3 seconds.
        """
        now = time.monotonic()
        cache_valid = (
            self._available is not None
            and (now - self._last_probe_at) < _LLM_PROBE_TTL
        )
        if cache_valid:
            return self._available  # type: ignore[return-value]

        if not _LLM_ENABLED:
            self._available = False
            self._last_probe_at = now
            return False
        try:
            import httpx
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=3.0)
            newly_available = resp.status_code == 200
            if newly_available and self._available is False:
                logger.info(
                    "LLMReasoner: Ollama back online at %s — switching to LLM reasoning.",
                    self.base_url,
                )
                self._llm = None  # Force reinitialisation of the client
            elif not newly_available:
                logger.warning(
                    "LLMReasoner: Ollama at %s returned HTTP %d.",
                    self.base_url, resp.status_code,
                )
            self._available = newly_available
        except Exception as exc:
            if self._available is not False:
                logger.info(
                    "LLMReasoner: Ollama not reachable at %s (%s). "
                    "Agents will use heuristic reasoning (re-probing every %.0fs).",
                    self.base_url, exc, _LLM_PROBE_TTL,
                )
            self._available = False
        self._last_probe_at = now
        return self._available  # type: ignore[return-value]

    def invoke(self, prompt: str) -> Optional[str]:
        """Call the LLM and return the raw text response, or None on failure.

        Args:
            prompt: Full ReAct prompt string.

        Returns:
            LLM response text, or None if unavailable/timed out.
        """
        if not self.is_available():
            return None
        llm = self._get_llm()
        if llm is None:
            return None
        try:
            response = llm.invoke(prompt)
            logger.debug("LLMReasoner.invoke: received %d chars.", len(response))
            return response
        except Exception as exc:
            logger.warning("LLMReasoner.invoke failed: %s", exc)
            self._available = False  # Mark as down for this session
            return None

    def _parse_json_from_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract the last JSON object from a ReAct-style response.

        The prompts instruct the LLM to output JSON on the last line.
        Falls back to searching the whole response for a JSON block.

        Args:
            response: Raw LLM response text.

        Returns:
            Parsed dict or None if no valid JSON found.
        """
        # Try last non-empty line first (most reliable).
        lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
        for line in reversed(lines):
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    pass

        # Fall back to regex extraction.
        match = re.search(r"\{[^{}]+\}", response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        logger.warning("LLMReasoner: could not parse JSON from response: %s…", response[:200])
        return None

    # ── Agent-specific reasoning methods ─────────────────────────────────────

    def classify_document(
        self,
        text: str,
        field_names: list,
        bart_doc_type: str = "unknown",
        bart_confidence: float = 0.5,
    ) -> Optional[Dict[str, Any]]:
        """Run LLM-based document classification.

        Args:
            text: Document text (truncated to 400 chars in prompt).
            field_names: NER field labels extracted from the document.
            bart_doc_type: Base classification from heuristic BART.
            bart_confidence: Confidence of the heuristic classification.

        Returns:
            Dict with ``doc_type``, ``confidence``, ``reasoning`` or None.
        """
        prompt = _CLASSIFIER_PROMPT.format(
            text=text[:400],
            field_names=", ".join(field_names) if field_names else "none",
            bart_doc_type=bart_doc_type,
            bart_confidence=bart_confidence,
        )
        response = self.invoke(prompt)
        if response is None:
            return None
        parsed = self._parse_json_from_response(response)
        if parsed:
            logger.info(
                "LLM classify: doc_type=%s confidence=%.2f",
                parsed.get("doc_type"), parsed.get("confidence", 0),
            )
        return parsed

    def review_extraction(
        self,
        doc_type: str,
        fields: list,
        missing_fields: list,
    ) -> Optional[Dict[str, Any]]:
        """Ask the LLM to validate extracted fields and flag uncertain ones.

        Args:
            doc_type: Classified document type.
            fields: List of extracted field dicts.
            missing_fields: Expected fields not found.

        Returns:
            Dict with ``overall_decision``, ``reliable_fields``,
            ``uncertain_fields``, ``reasoning`` or None.
        """
        fields_summary = json.dumps(
            [{"field": f.get("field_name"), "value": f.get("value"), "conf": round(f.get("confidence", 0), 2)}
             for f in fields[:8]],  # Cap to 8 fields to limit prompt length
            indent=None,
        )
        prompt = _EXTRACTOR_PROMPT.format(
            doc_type=doc_type,
            fields_json=fields_summary,
            missing_fields=", ".join(missing_fields) if missing_fields else "none",
        )
        response = self.invoke(prompt)
        if response is None:
            return None
        return self._parse_json_from_response(response)

    def assess_validation(
        self,
        vendor: str,
        amount: str,
        date: str,
        vendor_history: str,
        issues: list,
    ) -> Optional[Dict[str, Any]]:
        """Ask the LLM to assess validation risk and suggest confidence adjustment.

        Args:
            vendor: Extracted vendor name.
            amount: Extracted invoice amount.
            date: Extracted invoice date.
            vendor_history: Summary string from DB lookup.
            issues: List of validator issue descriptions.

        Returns:
            Dict with ``risk_level``, ``confidence_adjustment``,
            ``human_review_recommended``, ``reasoning`` or None.
        """
        issues_str = "; ".join(issues) if issues else "none"
        prompt = _VALIDATOR_PROMPT.format(
            vendor=vendor or "<unknown>",
            amount=amount or "<not found>",
            date=date or "<not found>",
            vendor_history=vendor_history or "no history found",
            issues=issues_str,
        )
        response = self.invoke(prompt)
        if response is None:
            return None
        return self._parse_json_from_response(response)

    def make_routing_decision(
        self,
        doc_type: str,
        extraction_confidence: float,
        is_valid: bool,
        vendor_known: bool,
        amount: str,
        safety_rails: list,
        risk_level: str = "medium",
        auto_threshold: float = 0.85,
        review_threshold: float = 0.65,
    ) -> Optional[Dict[str, Any]]:
        """Ask the LLM to make the final routing decision.

        Args:
            doc_type: Document type.
            extraction_confidence: Extractor confidence (0–1).
            is_valid: Whether validation passed.
            vendor_known: Whether vendor is in historical DB.
            amount: Extracted amount string.
            safety_rails: List of triggered safety rail identifiers.
            risk_level: Validator risk assessment.
            auto_threshold: Auto-approve confidence threshold.
            review_threshold: Human-review confidence threshold.

        Returns:
            Dict with ``action``, ``confidence``, ``reasoning`` or None.
        """
        prompt = _ROUTER_PROMPT.format(
            doc_type=doc_type,
            extraction_confidence=extraction_confidence,
            is_valid=is_valid,
            vendor_known=vendor_known,
            amount=amount or "<not found>",
            safety_rails=", ".join(safety_rails) if safety_rails else "none",
            risk_level=risk_level,
            auto_threshold=auto_threshold,
            review_threshold=review_threshold,
        )
        response = self.invoke(prompt)
        if response is None:
            return None
        return self._parse_json_from_response(response)


# ── Singleton ─────────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def get_reasoner() -> LLMReasoner:
    """Return the singleton LLMReasoner instance.

    Returns:
        Module-level LLMReasoner singleton.
    """
    return LLMReasoner()
