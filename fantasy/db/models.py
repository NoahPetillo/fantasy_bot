"""Multi-tenant ORM models (Postgres via SQLAlchemy).

User-owned rows carry a ``user_id`` foreign key so isolation is enforced at the
schema level (FK + unique constraints), not just in query code. ``snapshots`` is
the one exception: it is owned *transitively* through its ``league`` (per the
locked data model), so per-user snapshot queries join through ``leagues`` — see
the ``test_snapshot_read_isolation_via_league`` test. See MULTITENANT_BUILD.md →
"Data model" and hard requirement #1 (per-user isolation).

These tables are the destinations of the single-tenant → per-user migration:
``users``/``espn_credentials`` are new; ``leagues`` replaces ``data/leagues.json``;
``snapshots`` replaces the ``dashboard_<id>.json`` files; ``proposals`` replaces
the SQLite ``Store``; ``chat_usage`` backs the per-user chat quota. Phase 1 only
*defines* them — the code that reads/writes them lands in Phase 3.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone

from sqlalchemy import (
    BigInteger,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from fantasy.db.base import JSONColumn, GUID, Base, gen_uuid


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=gen_uuid)
    clerk_user_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    email: Mapped[str | None] = mapped_column(String(320), nullable=True)
    plan: Mapped[str] = mapped_column(String(32), nullable=False, default="free")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    espn_credential: Mapped[EspnCredential | None] = relationship(
        back_populates="user", uselist=False, cascade="all, delete-orphan"
    )
    leagues: Mapped[list[League]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class EspnCredential(Base):
    """One user's encrypted ESPN session cookies. ``user_id`` is the primary key
    (one row per user). ``s2_enc``/``swid_enc`` are Fernet ciphertext — the app
    NEVER stores or logs the plaintext cookies. See hard requirement #2."""

    __tablename__ = "espn_credentials"

    user_id: Mapped[uuid.UUID] = mapped_column(
        GUID, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    s2_enc: Mapped[str] = mapped_column(Text, nullable=False)
    swid_enc: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")
    # Count of consecutive ESPN auth failures; credentials auto-purge past a
    # threshold (hard requirement #2).
    failure_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    consent_version: Mapped[str] = mapped_column(String(32), nullable=False)
    consent_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    user: Mapped[User] = relationship(back_populates="espn_credential")


class League(Base):
    __tablename__ = "leagues"
    __table_args__ = (
        UniqueConstraint("user_id", "espn_league_id", "season", name="uq_league_user_espn_season"),
    )

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=gen_uuid)
    user_id: Mapped[uuid.UUID] = mapped_column(
        GUID, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    espn_league_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    team_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    season: Mapped[int] = mapped_column(Integer, nullable=False)
    name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    user: Mapped[User] = relationship(back_populates="leagues")
    snapshots: Mapped[list[Snapshot]] = relationship(
        back_populates="league", cascade="all, delete-orphan"
    )
    proposals: Mapped[list[Proposal]] = relationship(
        back_populates="league", cascade="all, delete-orphan"
    )


class Snapshot(Base):
    """A built dashboard payload for one league/week (replaces dashboard_<id>.json).
    Owned transitively via ``league`` — one row per (league, week), upserted."""

    __tablename__ = "snapshots"
    __table_args__ = (
        UniqueConstraint("league_id", "week", name="uq_snapshot_league_week"),
    )

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=gen_uuid)
    league_id: Mapped[uuid.UUID] = mapped_column(
        GUID, ForeignKey("leagues.id", ondelete="CASCADE"), nullable=False, index=True
    )
    week: Mapped[int | None] = mapped_column(Integer, nullable=True)
    payload: Mapped[dict] = mapped_column(JSONColumn, nullable=False, default=dict)
    built_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    league: Mapped[League] = relationship(back_populates="snapshots")


class Proposal(Base):
    """A recommended action awaiting approve/reject (replaces the SQLite Store).

    ``id`` is the domain proposal's stable hex id (the identifier the API and
    idempotency use). The full serialized domain proposal lives in ``payload``;
    the columns (kind/status/value/idempotency_key/league_id) are query indexes
    kept in sync. ``idempotency_key`` is unique *per user* so re-running a build
    never logs the same action twice — the store invariant, now scoped per user.
    """

    __tablename__ = "proposals"
    __table_args__ = (
        UniqueConstraint("user_id", "idempotency_key", name="uq_proposal_user_idempotency"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    user_id: Mapped[uuid.UUID] = mapped_column(
        GUID, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    league_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID, ForeignKey("leagues.id", ondelete="CASCADE"), nullable=True, index=True
    )
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="proposed")
    value: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    idempotency_key: Mapped[str | None] = mapped_column(String(64), nullable=True)
    payload: Mapped[dict] = mapped_column(JSONColumn, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    league: Mapped[League | None] = relationship(back_populates="proposals")


class ChatUsage(Base):
    """Per-user daily chat counter that backs the plan-based quota (Phase 5)."""

    __tablename__ = "chat_usage"

    user_id: Mapped[uuid.UUID] = mapped_column(
        GUID, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    day: Mapped[date] = mapped_column(Date, primary_key=True)
    count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
