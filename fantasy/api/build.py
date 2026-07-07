"""Snapshot building for one league — used by the CLI and the API's background
build thread. Cheap "shell" (settings + standings, instant) vs. full (model +
recommendations + report card)."""

from __future__ import annotations

import dataclasses
import logging

from fantasy.api.dashboard_data import assemble, shell_snapshot, write_snapshot
from fantasy.espn.client import EspnClient
from fantasy.leagues import LeagueRef, registry
from fantasy.orchestrator.store import Store

log = logging.getLogger(__name__)


def _client_and_settings(ref: LeagueRef):
    client = EspnClient(league_id=ref.league_id, season=ref.season)
    league = client.league_settings()
    # Backfill the league's display name into the registry the first time we see it.
    if not ref.name and getattr(league, "name", None):
        registry().add(dataclasses.replace(ref, name=league.name))
    return client, league


def _pick_week(client, requested: int | None) -> int:
    cur = int(getattr(client.league(), "current_week", 1) or 1)
    return requested or max(1, min(cur, 17))


def build_shell(ref: LeagueRef, week: int | None = None) -> dict:
    """Instant: just settings + standings + your team. No model."""
    client, league = _client_and_settings(ref)
    payload = shell_snapshot(client, league, ref.season, _pick_week(client, week), ref.team_id)
    write_snapshot(payload, ref.league_id)
    return payload


def build_full(ref: LeagueRef, week: int | None = None) -> dict:
    """Heavy: train the model + assemble recommendations + report card."""
    from fantasy.projections.service import ProjectionService, default_train_seasons

    client, league = _client_and_settings(ref)
    wk = _pick_week(client, week)
    log.info("Building full snapshot for league %s wk%s ...", ref.league_id, wk)
    service = ProjectionService(league).fit(default_train_seasons(ref.season))
    payload = assemble(service, league, Store(), ref.season, wk, client=client,
                       my_team_id=ref.team_id)
    write_snapshot(payload, ref.league_id)
    return payload
