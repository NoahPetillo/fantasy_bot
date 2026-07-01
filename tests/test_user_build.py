"""Per-user add-league endpoint: validates via the user's ESPN cookies (stubbed
here), persists on success, rolls back on failure."""

from __future__ import annotations

import fantasy.api.app as api
from fantasy.db.repos import list_leagues
from fantasy.espn.client import EspnAuthError


def test_add_league_creates_and_returns(webapp, monkeypatch):
    monkeypatch.setattr(api, "build_shell_for", lambda db, user, lg, week=None: {"ok": True})
    user = webapp.make_user("owner")
    webapp.auth_as(user)
    r = webapp.client.post("/api/leagues", json={"league_id": 777, "team_id": 3, "season": 2025})
    assert r.status_code == 200, r.text
    lg = r.json()["league"]
    assert lg["espn_league_id"] == 777 and lg["team_id"] == 3 and lg["season"] == 2025
    webapp.db.expire_all()
    assert [l.espn_league_id for l in list_leagues(webapp.db, user)] == [777]


def test_add_league_rolls_back_when_espn_unreachable(webapp, monkeypatch):
    def boom(db, user, lg, week=None):
        raise EspnAuthError("no creds")
    monkeypatch.setattr(api, "build_shell_for", boom)
    user = webapp.make_user("owner")
    webapp.auth_as(user)
    r = webapp.client.post("/api/leagues", json={"league_id": 888, "team_id": 1, "season": 2025})
    assert r.status_code == 400
    webapp.db.expire_all()
    assert list_leagues(webapp.db, user) == []  # rolled back — nothing persisted


def test_add_league_requires_league_id(webapp):
    webapp.auth_as(webapp.make_user("owner"))
    assert webapp.client.post("/api/leagues", json={"team_id": 1}).status_code == 400
