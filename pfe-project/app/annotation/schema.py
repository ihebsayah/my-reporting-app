"""Annotation schema definitions for document extraction."""

import logging
from dataclasses import dataclass
from typing import Dict, List

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtractionField:
    """Represents a single field in the extraction schema.

    Attributes:
        name: Canonical field label used across annotation and ML pipelines.
        description: Business meaning of the field.
        required: Whether the field is required for downstream processing.
        repeating: Whether multiple spans may be annotated for the field.
        color: Suggested display color in Label Studio.
        patterns: Human-readable examples or detection hints.
        include_rules: Guidance describing what belongs inside the span.
        exclude_rules: Guidance describing what should stay outside the span.
    """

    name: str
    description: str
    required: bool
    repeating: bool
    color: str
    patterns: List[str]
    include_rules: List[str]
    exclude_rules: List[str]


DEFAULT_INVOICE_SCHEMA: List[ExtractionField] = [
    ExtractionField(
        name="INVOICE_ID",
        description="Primary invoice reference or identifier.",
        required=True,
        repeating=False,
        color="#D97706",
        patterns=["INV-2026-001", "Invoice #12345", "Facture N 2026-11"],
        include_rules=[
            "Include the identifier value and business prefix when it is part of the ID.",
            "Keep separator characters such as hyphens or slashes when present.",
        ],
        exclude_rules=[
            "Exclude generic labels like Invoice, Ref, or Number when they are not part of the ID.",
        ],
    ),
    ExtractionField(
        name="INVOICE_DATE",
        description="Document issue date.",
        required=True,
        repeating=False,
        color="#2563EB",
        patterns=["2026-03-29", "29/03/2026", "March 29, 2026"],
        include_rules=[
            "Include the full date span including day, month, and year.",
        ],
        exclude_rules=[
            "Exclude nearby labels such as Date or Issue Date.",
        ],
    ),
    ExtractionField(
        name="DUE_DATE",
        description="Invoice payment deadline.",
        required=False,
        repeating=False,
        color="#7C3AED",
        patterns=["Due 15/04/2026", "Payment before April 15, 2026"],
        include_rules=[
            "Annotate only the due date value.",
        ],
        exclude_rules=[
            "Exclude payment terms text that does not belong to the date itself.",
        ],
    ),
    ExtractionField(
        name="VENDOR_NAME",
        description="Supplier or issuing company name.",
        required=True,
        repeating=False,
        color="#059669",
        patterns=["Acme Supplies LLC", "Globex SARL"],
        include_rules=[
            "Include the full legal or displayed company name.",
            "Include suffixes such as LLC, SARL, or Inc when shown.",
        ],
        exclude_rules=[
            "Exclude addresses, tax IDs, and contact details unless merged into the visible company name.",
        ],
    ),
    ExtractionField(
        name="TOTAL_AMOUNT",
        description="Final invoice total payable.",
        required=True,
        repeating=False,
        color="#DC2626",
        patterns=["1,240.00 USD", "$1,240.00", "TND 480.500"],
        include_rules=[
            "Include the amount and currency symbol or code when present.",
        ],
        exclude_rules=[
            "Exclude descriptive labels such as Total or Net Amount.",
            "Exclude subtotals or tax-only amounts.",
        ],
    ),
    ExtractionField(
        name="TAX_AMOUNT",
        description="Tax value applied to the invoice total.",
        required=False,
        repeating=False,
        color="#EA580C",
        patterns=["VAT 19%: 235.60", "Tax 14.000 TND"],
        include_rules=[
            "Annotate the numeric tax amount and its currency when present.",
        ],
        exclude_rules=[
            "Exclude the tax rate unless it is embedded in the same token span and cannot be separated cleanly.",
        ],
    ),
]


def get_default_invoice_schema() -> List[ExtractionField]:
    """Return the default invoice-focused extraction schema.

    Returns:
        A copy-like list of extraction field definitions.
    """
    logger.info("Loading default invoice extraction schema.")
    return list(DEFAULT_INVOICE_SCHEMA)


def schema_as_dict(fields: List[ExtractionField]) -> List[Dict[str, object]]:
    """Serialize extraction fields into dictionaries.

    Args:
        fields: Field definitions to serialize.

    Returns:
        A list of dictionaries for JSON-friendly output.
    """
    logger.debug("Serializing %d extraction fields.", len(fields))
    return [
        {
            "name": field.name,
            "description": field.description,
            "required": field.required,
            "repeating": field.repeating,
            "color": field.color,
            "patterns": field.patterns,
            "include_rules": field.include_rules,
            "exclude_rules": field.exclude_rules,
        }
        for field in fields
    ]
