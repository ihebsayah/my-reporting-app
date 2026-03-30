"""Annotation guideline generation helpers."""

import logging
from typing import List

from app.annotation.schema import ExtractionField

logger = logging.getLogger(__name__)


def build_annotation_guidelines(fields: List[ExtractionField], project_name: str) -> str:
    """Generate Markdown annotation guidelines.

    Args:
        fields: Extraction fields to document.
        project_name: Human-readable project name.

    Returns:
        Markdown-formatted annotation guidelines.
    """
    logger.info(
        "Generating annotation guidelines for project '%s' with %d fields.",
        project_name,
        len(fields),
    )
    sections: List[str] = [
        f"# Annotation Guidelines: {project_name}",
        "",
        "## General Rules",
        "- Annotate at span level and keep boundaries as tight as possible.",
        "- Include complete field values when they span multiple tokens or lines.",
        "- Exclude surrounding labels, punctuation, and explanatory text unless they are part of the value.",
        "- When in doubt, flag the sample for review instead of guessing.",
        "",
        "## Field Rules",
    ]

    for field in fields:
        sections.extend(_field_section(field))

    sections.extend(
        [
            "",
            "## Quality Control",
            "- Review 10-20 overlapping documents across annotators each week.",
            "- Target Cohen's Kappa of 0.80 or higher for all required fields.",
            "- Update this guideline file whenever a new edge case changes annotation behavior.",
        ]
    )
    return "\n".join(sections)


def _field_section(field: ExtractionField) -> List[str]:
    """Build a Markdown section for one extraction field.

    Args:
        field: Field definition to document.

    Returns:
        Markdown lines describing field-specific rules.
    """
    requirement = "Required" if field.required else "Optional"
    cardinality = "Repeating" if field.repeating else "Single span"
    lines: List[str] = [
        "",
        f"### {field.name}",
        f"- Description: {field.description}",
        f"- Requirement: {requirement}",
        f"- Cardinality: {cardinality}",
        "- Examples:",
    ]
    lines.extend(f"  - {pattern}" for pattern in field.patterns)
    lines.append("- Include:")
    lines.extend(f"  - {rule}" for rule in field.include_rules)
    lines.append("- Exclude:")
    lines.extend(f"  - {rule}" for rule in field.exclude_rules)
    return lines
