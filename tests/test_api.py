"""Per-user API approval flow: health, list, approve/reject, and the legacy
Slack path (owner single-tenant, global store)."""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

import fantasy.api.app as api
from fantasy.db.proposal_store import PgProposalStore
from fantasy.orchestrator.models import Proposal, ProposalKind, ProposalStatus
from fantasy.orchestrator.store import Store


def _seed_proposal(store, **kw) -> Proposal:
    p = Proposal(kind=ProposalKind.trade, season=2024, week=5, team_id=1,
                 title="Trade A for B", value=12.0,
                 payload={"key_fields": {"give": "A", "get": "B"}, "give": "A", "get": "B"}, **kw)
    store.add(p)
    return p


def test_health():
    h = TestClient(api.app).get("/health").json()
    assert h["status"] == "ok" and "mode" in h


def test_list_is_user_scoped(webapp):
    user = webapp.make_user("owner")
    store = PgProposalStore(webapp.db, user.id)
    p = _seed_proposal(store)
    webapp.auth_as(user)
    items = webapp.client.get("/api/proposals", params={"status": "proposed"}).json()
    assert any(it["id"] == p.id for it in items)


def test_approve_transitions_status(webapp):
    user = webapp.make_user("owner")
    store = PgProposalStore(webapp.db, user.id)
    p = _seed_proposal(store)
    webapp.auth_as(user)
    r = webapp.client.post(f"/proposals/{p.id}/approve").json()
    assert r["status"] == "approved"
    webapp.db.expire_all()
    assert store.get(p.id).status == ProposalStatus.approved
    # Re-deciding is a no-op.
    r2 = webapp.client.post(f"/proposals/{p.id}/reject").json()
    assert r2["status"] == "approved"


def test_approve_does_not_touch_espn_execute(webapp, monkeypatch):
    """The per-user approve path is read-only to ESPN — it must never call the
    execute layer (hard requirement #3)."""
    import fantasy.execute.base as execute_base

    called = {"n": 0}
    monkeypatch.setattr(execute_base, "execute_approved",
                        lambda *a, **k: called.__setitem__("n", called["n"] + 1))
    user = webapp.make_user("owner")
    store = PgProposalStore(webapp.db, user.id)
    p = _seed_proposal(store)
    webapp.auth_as(user)
    webapp.client.post(f"/proposals/{p.id}/approve")
    assert called["n"] == 0


def _seed_trade_snapshot(webapp, user, espn_id=111):
    """A league whose latest snapshot carries a realistic `trade` block."""
    from fantasy.db.repos import add_league, save_snapshot

    lg = add_league(webapp.db, user, espn_league_id=espn_id, team_id=1, season=2024, name="L")
    block = {
        "my_team_id": 1, "remaining_weeks": 8, "team_count": 2,
        "roster_slots": {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1}, "bench_size": 4,
        "team_names": {"1": "Me", "2": "Rival"},
        "teams": {"1": ["a", "b", "c", "d", "e", "f", "g"],
                  "2": ["x", "y", "z", "w", "v", "u", "t"]},
        "players": {
            "a": {"name": "A", "pos": "QB", "proj": 22, "vor": 6, "team_id": 1},
            "b": {"name": "B", "pos": "RB", "proj": 20, "vor": 9, "team_id": 1},
            "c": {"name": "C", "pos": "RB", "proj": 16, "vor": 6, "team_id": 1},
            "d": {"name": "D", "pos": "WR", "proj": 17, "vor": 8, "team_id": 1},
            "e": {"name": "E", "pos": "WR", "proj": 15, "vor": 6, "team_id": 1},
            "f": {"name": "F", "pos": "TE", "proj": 11, "vor": 4, "team_id": 1},
            "g": {"name": "G", "pos": "WR", "proj": 6, "vor": -2, "team_id": 1},
            "x": {"name": "X", "pos": "RB", "proj": 21, "vor": 10, "team_id": 2},
            "y": {"name": "Y", "pos": "WR", "proj": 18, "vor": 9, "team_id": 2},
            "z": {"name": "Z", "pos": "WR", "proj": 9, "vor": 0, "team_id": 2},
            "w": {"name": "W", "pos": "QB", "proj": 19, "vor": 4, "team_id": 2},
            "v": {"name": "V", "pos": "TE", "proj": 9, "vor": 2, "team_id": 2},
            "u": {"name": "U", "pos": "RB", "proj": 7, "vor": -2, "team_id": 2},
            "t": {"name": "T", "pos": "WR", "proj": 7, "vor": -2, "team_id": 2},
        },
    }
    save_snapshot(webapp.db, lg.id, 5, {"trade": block})
    return lg


def test_analyze_trade_multiplayer_returns_lineup_impact(webapp):
    user = webapp.make_user("ta1")
    lg = _seed_trade_snapshot(webapp, user)
    webapp.auth_as(user)
    r = webapp.client.post("/api/analyze-trade",
                           json={"give": ["g"], "get": ["x"], "league": str(lg.id)}).json()
    assert "error" not in r and "lineup_delta" in r
    assert r["accept_prob"] is not None            # single counterparty (team 2)
    assert r["with_team"] == "Rival"
    assert any(p["id"] == "x" for p in r["get"])


def test_analyze_trade_rejects_player_you_dont_own(webapp):
    user = webapp.make_user("ta2")
    lg = _seed_trade_snapshot(webapp, user, espn_id=112)
    webapp.auth_as(user)
    r = webapp.client.post("/api/analyze-trade",
                           json={"give": ["x"], "get": ["y"], "league": str(lg.id)}).json()
    assert "error" in r                            # x is on the rival's roster


def test_analyze_trade_rejects_same_player_both_sides(webapp):
    user = webapp.make_user("ta3")
    lg = _seed_trade_snapshot(webapp, user, espn_id=113)
    webapp.auth_as(user)
    r = webapp.client.post("/api/analyze-trade",
                           json={"give": ["b"], "get": ["b"], "league": str(lg.id)}).json()
    assert "error" in r


def test_analyze_trade_stale_snapshot_prompts_build(webapp):
    from fantasy.db.repos import add_league, save_snapshot

    user = webapp.make_user("ta4")
    webapp.auth_as(user)
    lg = add_league(webapp.db, user, espn_league_id=114, team_id=1, season=2024, name="L2")
    save_snapshot(webapp.db, lg.id, 5, {"board_index": {}})  # legacy payload, no trade block
    r = webapp.client.post("/api/analyze-trade",
                           json={"give": ["b"], "get": ["x"], "league": str(lg.id)}).json()
    assert "error" in r and "build" in r["error"].lower()


def test_slack_interaction_rejects_via_legacy_store():
    """Slack is the owner's single-tenant channel — it still uses the global store."""
    db = Path(tempfile.mkdtemp()) / "slack.sqlite"
    api._store = Store(db)
    p = _seed_proposal(api._store)
    client = TestClient(api.app)
    payload = '{"actions":[{"action_id":"reject_proposal","value":"%s"}]}' % p.id
    r = client.post("/slack/interactions", data={"payload": payload}).json()
    assert r["status"] == "rejected"
    assert api._store.get(p.id).status == ProposalStatus.rejected
    api._store = None
