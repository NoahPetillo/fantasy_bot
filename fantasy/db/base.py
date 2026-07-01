"""SQLAlchemy foundation for the multi-tenant data layer.

The engine is created lazily from :attr:`fantasy.config.Settings.effective_database_url`
so importing this module never requires a live database (imports stay cheap and
tests can point it at their own SQLite/Postgres). Production uses Neon Postgres;
dev/CI fall back to a local SQLite file.

Two portable column types (:class:`GUID`, :data:`JSONColumn`) let the *same*
models run on Postgres (native ``UUID``/``JSONB``) and SQLite (``CHAR(36)``/
``JSON``), which keeps the test suite fast and hermetic without a Postgres server.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator

from sqlalchemy import JSON, CHAR, create_engine, event
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.types import TypeDecorator

from fantasy.config import settings


@event.listens_for(Engine, "connect")
def _sqlite_fk_pragma(dbapi_conn, _conn_record):
    """Enforce foreign keys on SQLite (off by default; Postgres always enforces).
    Without this the FK/cascade guarantees behind per-user isolation wouldn't hold
    on the dev/CI SQLite fallback."""
    import sqlite3

    if isinstance(dbapi_conn, sqlite3.Connection):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()


class Base(DeclarativeBase):
    """Declarative base for every ORM model."""


class GUID(TypeDecorator):
    """Platform-independent UUID column.

    Uses PostgreSQL's native ``UUID`` when available, otherwise a ``CHAR(36)``
    holding the stringified value. Always yields :class:`uuid.UUID` in Python.
    """

    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(UUID(as_uuid=True))
        return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if not isinstance(value, uuid.UUID):
            value = uuid.UUID(str(value))
        if dialect.name == "postgresql":
            return value
        return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


# JSONB on Postgres, plain JSON everywhere else.
JSONColumn = JSON().with_variant(JSONB, "postgresql")


def gen_uuid() -> uuid.UUID:
    return uuid.uuid4()


# ── lazy engine / session ────────────────────────────────────────────────────
_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def _make_engine(url: str) -> Engine:
    kwargs: dict = {"pool_pre_ping": True, "future": True}
    if url.startswith("sqlite"):
        # SQLite is single-connection by default; the API and background build
        # threads share the engine, so allow cross-thread use (Postgres pools
        # handle this natively and ignore this arg).
        kwargs["connect_args"] = {"check_same_thread": False}
    return create_engine(url, **kwargs)


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = _make_engine(settings.effective_database_url)
    return _engine


def get_sessionmaker() -> sessionmaker[Session]:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            bind=get_engine(), autoflush=False, autocommit=False,
            expire_on_commit=False, future=True,
        )
    return _SessionLocal


def get_db() -> Iterator[Session]:
    """FastAPI dependency: yields a request-scoped session and always closes it."""
    db = get_sessionmaker()()
    try:
        yield db
    finally:
        db.close()


def create_all() -> None:
    """Create all tables on the current engine. Used by tests and first-run dev;
    production schema is managed by Alembic migrations."""
    import fantasy.db.models  # noqa: F401  (register models on Base.metadata)

    Base.metadata.create_all(bind=get_engine())


def configure_engine(url: str) -> None:
    """Point the data layer at an explicit URL (tests). Disposes any prior engine."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = _make_engine(url)
    _SessionLocal = None


def reset_engine() -> None:
    """Drop the cached engine/sessionmaker so the next access rebuilds from settings."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None
