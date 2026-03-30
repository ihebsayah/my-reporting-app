"""Database engine and session helpers."""

import logging
from typing import Generator, Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings, get_settings
from app.database.base import Base

logger = logging.getLogger(__name__)

_ENGINE: Optional[Engine] = None
_SESSION_FACTORY: Optional[sessionmaker[Session]] = None
_ENGINE_URL: Optional[str] = None


def get_engine(settings: Optional[Settings] = None) -> Engine:
    """Return a cached SQLAlchemy engine.

    Args:
        settings: Optional settings override.

    Returns:
        Cached SQLAlchemy engine.
    """
    global _ENGINE, _ENGINE_URL
    resolved_settings = settings or get_settings()
    if _ENGINE is None or _ENGINE_URL != resolved_settings.database_url:
        connect_args = (
            {"check_same_thread": False}
            if resolved_settings.database_url.startswith("sqlite")
            else {}
        )
        _ENGINE = create_engine(
            resolved_settings.database_url,
            future=True,
            pool_pre_ping=True,
            connect_args=connect_args,
        )
        _ENGINE_URL = resolved_settings.database_url
        logger.info("Initialized database engine.")
    return _ENGINE


def get_session_factory(settings: Optional[Settings] = None) -> sessionmaker[Session]:
    """Return a cached SQLAlchemy session factory.

    Args:
        settings: Optional settings override.

    Returns:
        Configured session factory.
    """
    global _SESSION_FACTORY
    resolved_settings = settings or get_settings()
    if _SESSION_FACTORY is None or _ENGINE_URL != resolved_settings.database_url:
        _SESSION_FACTORY = sessionmaker(
            bind=get_engine(resolved_settings),
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
            class_=Session,
        )
    return _SESSION_FACTORY


def init_database(settings: Optional[Settings] = None) -> None:
    """Create known database tables.

    Args:
        settings: Optional settings override.
    """
    engine = get_engine(settings)
    Base.metadata.create_all(bind=engine)
    logger.info("Ensured database tables exist.")


def get_db_session(settings: Optional[Settings] = None) -> Generator[Session, None, None]:
    """Yield a managed database session.

    Args:
        settings: Optional settings override.

    Yields:
        SQLAlchemy session instance.
    """
    session = get_session_factory(settings)()
    try:
        yield session
    finally:
        session.close()
