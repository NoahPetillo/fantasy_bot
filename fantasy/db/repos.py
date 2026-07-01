"""Per-user repositories for leagues + snapshots.

Every accessor is scoped to the owning user (leagues) or reached only through the
user's leagues (snapshots), so one user can never read or mutate another's rows
(hard requirement #1). Leagues replace ``data/leagues.json``; snapshots replace
the ``dashboard_<id>.json`` files.
"""

from __future__ import annotations

import uuid as _uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from fantasy.db.models import League, Snapshot, User


def _as_uuid(value) -> _uuid.UUID | None:
    if isinstance(value, _uuid.UUID):
        return value
    try:
        return _uuid.UUID(str(value))
    except (ValueError, TypeError, AttributeError):
        return None


# ── leagues ──────────────────────────────────────────────────────────────────
def list_leagues(db: Session, user: User) -> list[League]:
    return list(db.execute(
        select(League).where(League.user_id == user.id).order_by(League.created_at)
    ).scalars().all())


def get_league(db: Session, user: User, league_uuid) -> League | None:
    """Fetch one league BY its internal id, scoped to the user — returns None if it
    doesn't exist or belongs to someone else."""
    uid = _as_uuid(league_uuid)
    if uid is None:
        return None
    return db.execute(
        select(League).where(League.id == uid, League.user_id == user.id)
    ).scalar_one_or_none()


def get_league_by_espn(db: Session, user: User, espn_league_id: int, season: int) -> League | None:
    return db.execute(
        select(League).where(League.user_id == user.id,
                             League.espn_league_id == espn_league_id,
                             League.season == season)
    ).scalar_one_or_none()


def add_league(db: Session, user: User, espn_league_id: int, team_id: int | None,
               season: int, name: str | None = None) -> League:
    """Upsert a league for this user (unique per user+espn_league_id+season)."""
    lg = get_league_by_espn(db, user, espn_league_id, season)
    if lg is None:
        lg = League(user_id=user.id, espn_league_id=espn_league_id,
                    team_id=team_id, season=season, name=name)
        db.add(lg)
    else:
        lg.team_id = team_id
        if name:
            lg.name = name
    db.commit()
    db.refresh(lg)
    return lg


def set_league_name(db: Session, league: League, name: str) -> None:
    if name and league.name != name:
        league.name = name
        db.commit()


def remove_league(db: Session, user: User, league_uuid) -> bool:
    lg = get_league(db, user, league_uuid)
    if lg is None:
        return False
    db.delete(lg)  # cascades to its snapshots + proposals
    db.commit()
    return True


# ── snapshots ────────────────────────────────────────────────────────────────
def save_snapshot(db: Session, league_id, week: int | None, payload: dict) -> Snapshot:
    """Upsert the built dashboard payload for (league, week)."""
    lid = _as_uuid(league_id)
    row = db.execute(
        select(Snapshot).where(Snapshot.league_id == lid, Snapshot.week == week)
    ).scalar_one_or_none()
    if row is None:
        row = Snapshot(league_id=lid, week=week, payload=payload)
        db.add(row)
    else:
        row.payload = payload
    db.commit()
    db.refresh(row)
    return row


def latest_snapshot(db: Session, league_id) -> dict | None:
    """Most recently built snapshot payload for a league (any week)."""
    lid = _as_uuid(league_id)
    row = db.execute(
        select(Snapshot).where(Snapshot.league_id == lid)
        .order_by(Snapshot.built_at.desc())
    ).scalars().first()
    return row.payload if row else None


def latest_snapshot_for_user(db: Session, user: User, league_uuid) -> dict | None:
    """Latest snapshot for one of the user's leagues, enforcing ownership."""
    lg = get_league(db, user, league_uuid)
    if lg is None:
        return None
    return latest_snapshot(db, lg.id)
