"""Logging helpers for the application."""

import logging
from typing import Optional

from app.config import Settings, get_settings


LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
)


def configure_logging(settings: Optional[Settings] = None) -> None:
    """Configure application-wide logging.

    Args:
        settings: Optional settings object. If omitted, cached settings are used.
    """
    resolved_settings = settings or get_settings()
    logging.basicConfig(level=resolved_settings.log_level.upper(), format=LOG_FORMAT)
