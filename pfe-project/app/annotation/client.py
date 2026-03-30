"""Minimal Label Studio API client for bootstrap, import, and export tasks."""

import logging
from typing import Any, Dict, List, Optional

import requests

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


class LabelStudioClient:
    """Simple client for creating projects and syncing tasks with Label Studio."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """Initialize the client.

        Args:
            settings: Optional application settings override.
        """
        self.settings = settings or get_settings()
        self.base_url = self.settings.label_studio_url.rstrip("/")
        self.headers = {
            "Authorization": f"Token {self.settings.label_studio_api_key}",
            "Content-Type": "application/json",
        }

    def create_project(self, title: str, label_config: str) -> Dict[str, Any]:
        """Create a Label Studio project.

        Args:
            title: Project title.
            label_config: XML label configuration.

        Returns:
            Parsed JSON response from Label Studio.

        Raises:
            requests.HTTPError: If the request fails.
        """
        logger.info("Creating Label Studio project '%s'.", title)
        payload = {"title": title, "label_config": label_config}
        response = requests.post(
            f"{self.base_url}/api/projects",
            json=payload,
            headers=self.headers,
            timeout=30,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            logger.error("Failed to create Label Studio project '%s': %s", title, exc)
            raise
        return response.json()

    def import_tasks(self, project_id: int, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Import labeling tasks into an existing Label Studio project.

        Args:
            project_id: Target Label Studio project ID.
            tasks: Task payloads to import.

        Returns:
            Parsed JSON response from Label Studio.

        Raises:
            requests.HTTPError: If the request fails.
        """
        logger.info(
            "Importing %d tasks into Label Studio project %d.", len(tasks), project_id
        )
        response = requests.post(
            f"{self.base_url}/api/projects/{project_id}/import",
            json=tasks,
            headers=self.headers,
            timeout=60,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            logger.error(
                "Failed to import tasks into Label Studio project %d: %s",
                project_id,
                exc,
            )
            raise
        return response.json()

    def export_annotations(
        self, project_id: int, export_type: str = "JSON"
    ) -> List[Dict[str, Any]]:
        """Export annotations from a Label Studio project.

        Args:
            project_id: Target Label Studio project ID.
            export_type: Label Studio export format identifier.

        Returns:
            Raw exported task payload.

        Raises:
            requests.HTTPError: If the request fails.
        """
        logger.info(
            "Exporting annotations from Label Studio project %d as %s.",
            project_id,
            export_type,
        )
        response = requests.get(
            f"{self.base_url}/api/projects/{project_id}/export",
            params={"exportType": export_type},
            headers=self.headers,
            timeout=60,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            logger.error(
                "Failed to export annotations from Label Studio project %d: %s",
                project_id,
                exc,
            )
            raise
        payload = response.json()
        if not isinstance(payload, list):
            logger.error("Expected a list export payload for project %d.", project_id)
            raise ValueError("Label Studio export payload must be a list of tasks.")
        return payload
