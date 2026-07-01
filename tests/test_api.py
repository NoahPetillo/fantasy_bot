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
