"""Tests for annotation schema and Label Studio config generation."""

import json

from app.annotation.guidelines import build_annotation_guidelines
from app.annotation.label_studio_config import LabelStudioConfigBuilder
from app.annotation.schema import get_default_invoice_schema, schema_as_dict


def test_default_invoice_schema_contains_required_fields() -> None:
    """Ensure the default schema exposes the expected core invoice fields."""
    fields = get_default_invoice_schema()

    field_names = [field.name for field in fields]

    assert "INVOICE_ID" in field_names
    assert "INVOICE_DATE" in field_names
    assert "VENDOR_NAME" in field_names
    assert "TOTAL_AMOUNT" in field_names
    assert len(fields) >= 5


def test_schema_serialization_is_json_friendly() -> None:
    """Ensure schema serialization can be dumped to JSON cleanly."""
    payload = schema_as_dict(get_default_invoice_schema())

    dumped = json.dumps(payload)

    assert "INVOICE_ID" in dumped
    assert "required" in dumped


def test_label_studio_config_contains_labels_and_text_binding() -> None:
    """Ensure the generated XML includes the expected Label Studio bindings."""
    fields = get_default_invoice_schema()
    builder = LabelStudioConfigBuilder("Invoice Project", fields)

    xml_config = builder.build_xml()

    assert "<View>" in xml_config
    assert "<Text name=\"text\" value=\"$text\"" in xml_config
    assert 'value="INVOICE_ID"' in xml_config
    assert 'value="TOTAL_AMOUNT"' in xml_config


def test_annotation_guidelines_capture_general_and_field_rules() -> None:
    """Ensure generated guidelines contain reusable annotation instructions."""
    guidelines = build_annotation_guidelines(
        fields=get_default_invoice_schema(),
        project_name="Invoice Project",
    )

    assert "# Annotation Guidelines: Invoice Project" in guidelines
    assert "## General Rules" in guidelines
    assert "### INVOICE_ID" in guidelines
    assert "Cohen's Kappa" in guidelines
