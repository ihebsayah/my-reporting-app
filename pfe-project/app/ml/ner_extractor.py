"""Regex + spaCy ensemble NER extraction."""

import importlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Pattern, Sequence, Tuple

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtractedEntity:
    """Represents one extracted entity span."""

    start: int
    end: int
    text: str
    label: str
    sources: Tuple[str, ...]
    score: float


@dataclass(frozen=True)
class ExtractionResult:
    """Represents ensemble extraction results for a document."""

    text: str
    entities: List[ExtractedEntity] = field(default_factory=list)


class RegexSpacyEnsembleExtractor:
    """Combine deterministic regex matches with optional spaCy NER predictions."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        spacy_model: Any = None,
    ) -> None:
        """Initialize the ensemble extractor.

        Args:
            settings: Optional application settings override.
            spacy_model: Optional injected spaCy pipeline for testing or custom usage.
        """
        self.settings = settings or get_settings()
        self._spacy_model = spacy_model
        self._patterns = self._build_patterns()

    def extract(self, text: str) -> ExtractionResult:
        """Extract entities from text using regex and optional spaCy NER.

        Args:
            text: Raw document text.

        Returns:
            Ensemble extraction result.
        """
        regex_entities = self._extract_with_regex(text)
        spacy_entities = self._extract_with_spacy(text)
        merged_entities = self._merge_entities(regex_entities, spacy_entities)
        logger.info(
            "Extracted %d entities (%d regex, %d spaCy).",
            len(merged_entities),
            len(regex_entities),
            len(spacy_entities),
        )
        return ExtractionResult(text=text, entities=merged_entities)

    def _extract_with_regex(self, text: str) -> List[ExtractedEntity]:
        """Extract deterministic entities using regex patterns."""
        entities: List[ExtractedEntity] = []
        for label, patterns in self._patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    value = match.group(1) if match.lastindex else match.group(0)
                    start, end = self._value_offsets(match, value)
                    entities.append(
                        ExtractedEntity(
                            start=start,
                            end=end,
                            text=value,
                            label=label,
                            sources=("regex",),
                            score=0.85,
                        )
                    )
        return self._deduplicate_entities(entities)

    def _extract_with_spacy(self, text: str) -> List[ExtractedEntity]:
        """Extract entities using a spaCy model when available."""
        model = self._load_spacy_model()
        if model is None:
            return []

        doc = model(text)
        entities: List[ExtractedEntity] = []
        for entity in getattr(doc, "ents", []):
            entities.append(
                ExtractedEntity(
                    start=int(entity.start_char),
                    end=int(entity.end_char),
                    text=str(entity.text),
                    label=str(entity.label_),
                    sources=("spacy",),
                    score=0.75,
                )
            )
        return self._deduplicate_entities(entities)

    def _load_spacy_model(self) -> Any:
        """Load the configured spaCy model lazily."""
        if self._spacy_model is not None:
            return self._spacy_model

        model_path = Path(self.settings.ner_model_path)
        if not model_path.exists():
            logger.info("spaCy model path does not exist yet: %s", model_path)
            return None

        try:
            spacy_module = importlib.import_module("spacy")
        except ImportError:
            logger.warning("spaCy is not installed; ensemble will use regex-only mode.")
            return None

        self._spacy_model = spacy_module.load(model_path)
        logger.info("Loaded spaCy NER model from %s.", model_path)
        return self._spacy_model

    def _merge_entities(
        self,
        regex_entities: Sequence[ExtractedEntity],
        spacy_entities: Sequence[ExtractedEntity],
    ) -> List[ExtractedEntity]:
        """Merge regex and spaCy entities by exact span and label."""
        merged: Dict[Tuple[int, int, str], ExtractedEntity] = {}
        for entity in list(regex_entities) + list(spacy_entities):
            key = (entity.start, entity.end, entity.label)
            if key not in merged:
                merged[key] = entity
                continue
            existing = merged[key]
            sources = tuple(sorted(set(existing.sources + entity.sources)))
            merged[key] = ExtractedEntity(
                start=entity.start,
                end=entity.end,
                text=entity.text,
                label=entity.label,
                sources=sources,
                score=max(existing.score, entity.score) + (0.1 if len(sources) > 1 else 0.0),
            )
        return sorted(merged.values(), key=lambda item: (item.start, item.end, item.label))

    def _deduplicate_entities(
        self, entities: Sequence[ExtractedEntity]
    ) -> List[ExtractedEntity]:
        """Drop duplicate entities within one source."""
        unique: Dict[Tuple[int, int, str], ExtractedEntity] = {}
        for entity in entities:
            unique[(entity.start, entity.end, entity.label)] = entity
        return sorted(unique.values(), key=lambda item: (item.start, item.end, item.label))

    def _build_patterns(self) -> Dict[str, List[Pattern[str]]]:
        """Build regex patterns for common invoice fields."""
        return {
            "INVOICE_ID": [
                re.compile(r"\b(INV[-/ ]?\d{3,8}(?:[-/]\d{2,8})?)\b", re.IGNORECASE),
                re.compile(r"Invoice\s*#?\s*([A-Z0-9-]{3,20})", re.IGNORECASE),
            ],
            "INVOICE_DATE": [
                re.compile(
                    r"\b(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|[A-Z][a-z]+\s+\d{1,2},\s+\d{4})\b"
                )
            ],
            "TOTAL_AMOUNT": [
                re.compile(r"Total:\s*([$€£]?\d[\d,]*\.\d{2})", re.IGNORECASE),
                re.compile(r"\b([$€£]\d[\d,]*\.\d{2})\b"),
                re.compile(r"\b(\d[\d,]*\.\d{2}\s?(?:USD|EUR|TND))\b", re.IGNORECASE),
            ],
            "VENDOR_NAME": [
                re.compile(
                    r"Vendor:\s*([A-Z][A-Za-z0-9&.,' -]*(?:LLC|SARL|Inc|Ltd|Corp|Company|Traders)?)",
                    re.IGNORECASE,
                )
            ],
        }

    @staticmethod
    def _value_offsets(match: re.Match[str], value: str) -> Tuple[int, int]:
        """Return absolute offsets for the extracted value within the full text."""
        start = match.start(1) if match.lastindex else match.start()
        end = start + len(value)
        return start, end
