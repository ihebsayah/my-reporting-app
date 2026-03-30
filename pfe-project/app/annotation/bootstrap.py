"""Month 1 bootstrap helpers for the annotation workflow."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from app.annotation.guidelines import build_annotation_guidelines
from app.annotation.label_studio_config import LabelStudioConfigBuilder
from app.annotation.schema import ExtractionField, get_default_invoice_schema, schema_as_dict
from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


def build_annotation_bootstrap_artifacts(
    output_dir: Path,
    settings: Optional[Settings] = None,
    fields: Optional[List[ExtractionField]] = None,
) -> Dict[str, str]:
    """Create the initial annotation bootstrap files on disk.

    Args:
        output_dir: Directory where bootstrap artifacts will be written.
        settings: Optional application settings override.
        fields: Optional field definitions override.

    Returns:
        Paths to the generated artifacts as strings.
    """
    resolved_settings = settings or get_settings()
    resolved_fields = fields or get_default_invoice_schema()
    logger.info("Writing annotation bootstrap artifacts to %s.", output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    schema_path = output_dir / "invoice_schema.json"
    config_path = output_dir / "label_studio_config.xml"
    guidelines_path = output_dir / "annotation_guidelines.md"

    builder = LabelStudioConfigBuilder(
        project_title=resolved_settings.label_studio_project_title,
        fields=resolved_fields,
    )
    guidelines = build_annotation_guidelines(
        fields=resolved_fields,
        project_name=resolved_settings.label_studio_project_title,
    )

    schema_path.write_text(
        json.dumps(schema_as_dict(resolved_fields), indent=2),
        encoding="utf-8",
    )
    config_path.write_text(builder.build_xml(), encoding="utf-8")
    guidelines_path.write_text(guidelines, encoding="utf-8")

    return {
        "schema": str(schema_path),
        "label_config": str(config_path),
        "guidelines": str(guidelines_path),
    }
