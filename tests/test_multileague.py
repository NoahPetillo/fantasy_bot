"""Multi-league registry + decision tracking (undo/confirm/influence)."""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

import fantasy.api.app as api
from fantasy.leagues import LeagueRef, LeagueRegistry
from fantasy.orchestrator.influence import influence_stats
from fantasy.orchestrator.models import Proposal, ProposalKind, ProposalStatus
from fantasy.orchestrator.store import Store


# ── registry ──────────────────────────────────────────────────────────────────
def test_registry_add_get_remove_upsert(tmp_path):
    reg = LeagueRegistry(tmp_path / "leagues.json")
    reg.add(LeagueRef(league_id=111, team_id=2, season=2025, name="A"))
    reg.add(LeagueRef(league_id=222, team_id=3, season=2026))
    assert {r.league_id for r in reg.all()} == {111, 222}
    reg.add(LeagueRef(league_id=111, team_id=9, season=2025, name="A2"))  # upsert by id
    assert reg.get(111).team_id == 9 and reg.get(111).name == "A2"
    assert len(reg.all()) == 2
    assert reg.remove(222) is True and reg.get(222) is None
    assert reg.remove(999) is False


# ── influence ledger ──────────────────────────────────────────────────────────
def _seed_store() -> Store:
    s = Store(Path(tempfile.mkdtemp()) / "inf.sqlite")

    def mk(kind, val):
        return Proposal(kind=kind, season=2025, week=3, team_id=7,
                        title=kind.value, value=val, payload={"key_fields": {"k": val}})

    a, b, c, d = (mk(ProposalKind.trade, 5), mk(ProposalKind.waiver, 4),
                  mk(ProposalKind.trade, 3), mk(ProposalKind.alert, 1))
    for p in (a, b, c, d):
        s.add(p)
    s.set_status(a.id, ProposalStatus.approved)
    s.set_status(b.id, ProposalStatus.executed)   # confirmed
    s.set_status(c.id, ProposalStatus.rejected)
    return s


def test_influence_stats_counts_and_rate():
    st = influence_stats(_seed_store(), season=2025, team_id=7)
    assert st["followed"] == 2 and st["confirmed"] == 1 and st["rejected"] == 1
    assert st["pending"] == 0 and st["total"] == 3      # the alert is excluded
    assert st["acceptance_rate"] == round(2 / 3, 2)
    assert st["by_kind"]["trade"] == 2


# ── decision endpoints ────────────────────────────────────────────────────────
def _client_with_seed():
    db = Path(tempfile.mkdtemp()) / "api.sqlite"
    api._store = Store(db)
    p = Proposal(kind=ProposalKind.trade, season=2024, week=5, team_id=1,
                 title="Trade A for B", value=12.0,
                 payload={"key_fields": {"give": "A", "get": "B"}, "give": "A", "get": "B"})
    api._store.add(p)
    return TestClient(api.app), p


def test_undo_reverts_to_proposed():
    client, p = _client_with_seed()
    client.post(f"/proposals/{p.id}/approve")
    r = client.post(f"/proposals/{p.id}/undo").json()
    assert r["status"] == "proposed"
    assert api._store.get(p.id).status == ProposalStatus.proposed


def test_confirm_without_league_is_unconfirmed():
    client, p = _client_with_seed()
    client.post(f"/proposals/{p.id}/approve")
    r = client.post(f"/proposals/{p.id}/confirm").json()
    assert r["confirmed"] is False
    assert "No league" in r["detail"]
    # still approved (not executed) since it couldn't be confirmed
    assert api._store.get(p.id).status == ProposalStatus.approved


def test_leagues_endpoint_lists(monkeypatch, tmp_path):
    reg = LeagueRegistry(tmp_path / "leagues.json")
    reg.add(LeagueRef(league_id=555, team_id=1, season=2025, name="Z"))
    monkeypatch.setattr(api, "registry", lambda: reg)
    client = TestClient(api.app)
    d = client.get("/api/leagues").json()
    assert any(lg["league_id"] == 555 for lg in d["leagues"])
    assert d["active"] == 555
