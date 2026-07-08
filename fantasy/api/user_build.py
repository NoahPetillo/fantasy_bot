"""Per-user snapshot building (Postgres-backed).

The multi-tenant equivalent of ``fantasy/api/build.py``: reads the user's league
with THEIR decrypted ESPN cookies, runs the same decision layer (``assemble`` /
``shell_snapshot``), persists proposals to the per-user store, and saves the
payload to the ``snapshots`` table. Read-only to ESPN throughout.
"""

from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from fantasy.api.dashboard_data import assemble, shell_snapshot
from fantasy.db.models import League, User
from fantasy.db.proposal_store import PgProposalStore
from fantasy.db.repos import save_snapshot, set_league_name
from fantasy.espn.credentials import build_client_for_user
from fantasy.league_rules import effective_settings

log = logging.getLogger(__name__)


def _pick_week(client, requested: int | None) -> int:
    cur = int(getattr(client.league(), "current_week", 1) or 1)
    return requested or max(1, min(cur, 17))


def build_shell_for(db: Session, user: User, league: League, week: int | None = None) -> dict:
    """Instant shell (settings + standings + team), stored per-user."""
    client = build_client_for_user(db, user, league.espn_league_id, league.season)
    ls = effective_settings(db, league, client)
    set_league_name(db, league, getattr(ls, "name", "") or "")
    wk = _pick_week(client, week)
    payload = shell_snapshot(client, ls, league.season, wk, league.team_id)
    save_snapshot(db, league.id, wk, payload)
    return payload


def build_full_for(db: Session, user: User, league: League, week: int | None = None) -> dict:
    """Heavy build (model + recommendations + report card), stored per-user."""
    from fantasy.projections.service import ProjectionService, default_train_seasons

    client = build_client_for_user(db, user, league.espn_league_id, league.season)
    ls = effective_settings(db, league, client)
    set_league_name(db, league, getattr(ls, "name", "") or "")
    wk = _pick_week(client, week)
    log.info("Building full snapshot: user=%s espn_league=%s wk=%s",
             user.id, league.espn_league_id, wk)
    service = ProjectionService(ls).fit(default_train_seasons(league.season))
    store = PgProposalStore(db, user.id, league.id)
    payload = assemble(service, ls, store, league.season, wk, client=client,
                       my_team_id=league.team_id)
    save_snapshot(db, league.id, wk, payload)
    return payload
