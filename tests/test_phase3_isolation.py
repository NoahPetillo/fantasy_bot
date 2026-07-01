"""Hard requirement #1 — per-user isolation across EVERY user-facing endpoint.

User B must not be able to read or mutate user A's leagues, snapshots, or
proposals through any route. Data is seeded directly for A, then every endpoint is
exercised as B (denied) and as A (allowed).
"""

from __future__ import annotations

import pytest

from fantasy.db.proposal_store import PgProposalStore
from fantasy.db.repos import add_league, save_snapshot
from fantasy.orchestrator.models import Proposal, ProposalKind, ProposalStatus


@pytest.fixture
def two_users(webapp):
    db = webapp.db
    a = webapp.make_user("alice")
    b = webapp.make_user("bob")
    la = add_league(db, a, espn_league_id=111, team_id=1, season=2025, name="Alice League")
    store_a = PgProposalStore(db, a.id, la.id)
    pa = Proposal(kind=ProposalKind.trade, season=2025, week=5, team_id=1,
                  title="Alice trade", value=10.0,
                  payload={"key_fields": {"give": "X", "get": "Y"}, "give": "X", "get": "Y"})
    store_a.add(pa)
    save_snapshot(db, la.id, 5, {
        "team": {"season": 2025, "week": 5, "team_id": 1, "name": "Alice"},
        "actions": [{"id": pa.id, "kind": "trade", "title": "Alice trade", "value": 10.0,
                     "status": "proposed"}],
    })
    return webapp, a, b, la, pa, store_a


# ── B is denied A's data on every endpoint ──────────────────────────────────────
def test_b_cannot_list_or_touch_a_proposals(two_users):
    webapp, a, b, la, pa, store_a = two_users
    webapp.auth_as(b)
    c = webapp.client
    assert c.get("/api/proposals").json() == []                        # not visible
    assert c.post(f"/proposals/{pa.id}/approve").status_code == 404     # can't approve
    assert c.post(f"/proposals/{pa.id}/reject").status_code == 404
    assert c.post(f"/proposals/{pa.id}/undo").status_code == 404
    assert c.post(f"/proposals/{pa.id}/confirm").status_code == 404
    webapp.db.expire_all()
    assert store_a.get(pa.id).status == ProposalStatus.proposed        # untouched


def test_b_cannot_see_or_mutate_a_leagues(two_users):
    webapp, a, b, la, pa, store_a = two_users
    webapp.auth_as(b)
    c = webapp.client
    assert c.get("/api/leagues").json() == {"leagues": [], "active": None}
    assert c.delete(f"/api/leagues/{la.id}").status_code == 404
    assert c.post(f"/api/leagues/{la.id}/build").status_code == 404
    assert c.get(f"/api/dashboard?league={la.id}").status_code == 404


def test_b_own_dashboard_is_empty(two_users):
    webapp, a, b, la, pa, store_a = two_users
    webapp.auth_as(b)
    d = webapp.client.get("/api/dashboard").json()
    assert d["team"]["name"] == "No league yet" and d["actions"] == []


# ── A sees and controls A's own data ────────────────────────────────────────────
def test_a_sees_and_controls_own_data(two_users):
    webapp, a, b, la, pa, store_a = two_users
    webapp.auth_as(a)
    c = webapp.client
    assert any(it["id"] == pa.id for it in c.get("/api/proposals").json())
    dash = c.get(f"/api/dashboard?league={la.id}").json()
    assert dash["statuses"].get(pa.id) == "proposed"
    assert c.post(f"/proposals/{pa.id}/approve").json()["status"] == "approved"
    webapp.db.expire_all()
    assert store_a.get(pa.id).status == ProposalStatus.approved
    # …and B still cannot after A acted.
    webapp.auth_as(b)
    assert webapp.client.post(f"/proposals/{pa.id}/reject").status_code == 404
