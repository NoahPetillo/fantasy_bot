"""API approval flow: list, approve, reject, Slack interaction."""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

import fantasy.api.app as api
from fantasy.orchestrator.models import Proposal, ProposalKind, ProposalStatus
from fantasy.orchestrator.store import Store


def _client_with_seed():
    db = Path(tempfile.mkdtemp()) / "api.sqlite"
    api._store = Store(db)  # point the app at a fresh store
    p = Proposal(kind=ProposalKind.trade, season=2024, week=5, team_id=1,
                 title="Trade A for B", value=12.0,
                 payload={"key_fields": {"give": "A", "get": "B"}})
    api._store.add(p)
    return TestClient(api.app), p


def test_health_and_list():
    client, p = _client_with_seed()
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["pending_proposals"] >= 1
    items = client.get("/api/proposals", params={"status": "proposed"}).json()
    assert any(it["id"] == p.id for it in items)


def test_approve_transitions_status():
    client, p = _client_with_seed()
    r = client.post(f"/proposals/{p.id}/approve").json()
    assert r["status"] == "approved"
    assert api._store.get(p.id).status == ProposalStatus.approved
    # Re-deciding is a no-op.
    r2 = client.post(f"/proposals/{p.id}/reject").json()
    assert r2["status"] == "approved"


def test_slack_interaction_rejects():
    client, p = _client_with_seed()
    payload = (
        '{"actions":[{"action_id":"reject_proposal","value":"%s"}]}' % p.id
    )
    r = client.post("/slack/interactions", data={"payload": payload}).json()
    assert r["status"] == "rejected"
    assert api._store.get(p.id).status == ProposalStatus.rejected
