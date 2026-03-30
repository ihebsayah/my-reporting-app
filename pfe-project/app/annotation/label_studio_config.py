"""Utilities for generating Label Studio project configuration."""

import logging
from typing import List
from xml.sax.saxutils import escape

from app.annotation.schema import ExtractionField

logger = logging.getLogger(__name__)


class LabelStudioConfigBuilder:
    """Build Label Studio XML configuration for span-level annotation projects."""

    def __init__(self, project_title: str, fields: List[ExtractionField]) -> None:
        """Initialize the builder.

        Args:
            project_title: Human-readable project title.
            fields: Extraction fields to expose as labels.
        """
        self.project_title = project_title
        self.fields = fields

    def build_xml(self) -> str:
        """Generate XML label configuration for Label Studio.

        Returns:
            XML configuration as a string.
        """
        logger.info(
            "Building Label Studio XML config for project '%s' with %d fields.",
            self.project_title,
            len(self.fields),
        )
        labels = "\n".join(self._build_label_tag(field) for field in self.fields)
        title = escape(self.project_title)
        return (
            f"<View>\n"
            f"  <Header value=\"{title}\"/>\n"
            "  <Text name=\"text\" value=\"$text\" granularity=\"word\"/>\n"
            "  <Labels name=\"label\" toName=\"text\">\n"
            f"{labels}\n"
            "  </Labels>\n"
            "</View>"
        )

    @staticmethod
    def _build_label_tag(field: ExtractionField) -> str:
        """Create a Label Studio label tag for one extraction field.

        Args:
            field: Field definition to convert.

        Returns:
            XML fragment for a Label Studio label.
        """
        return (
            "    <Label "
            f"value=\"{escape(field.name)}\" "
            f"background=\"{escape(field.color)}\" "
            f"hint=\"{escape(field.description)}\"/>"
        )
