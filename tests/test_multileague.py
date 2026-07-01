"""Legacy registry + influence ledger (domain units) and the per-user decision
endpoints (undo / confirm / leagues)."""

from __future__ import annotations

import tempfile
from pathlib import Path

from fantasy.db.proposal_store import PgProposalStore
from fantasy.db.repos import add_league
from fantasy.leagues import LeagueRef, LeagueRegistry
from fantasy.orchestrator.influence import influence_stats
from fantasy.orchestrator.models import Proposal, ProposalKind, ProposalStatus
from fantasy.orchestrator.store import Store


# ── registry (legacy domain object, still used by scripts/orchestrator) ─────────
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


# ── influence ledger (works over any store, here the legacy SQLite Store) ───────
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


# ── per-user decision endpoints ─────────────────────────────────────────────────
def _seed_proposal(store, payload=None) -> Proposal:
    p = Proposal(kind=ProposalKind.trade, season=2024, week=5, team_id=1,
                 title="Trade A for B", value=12.0,
                 payload=payload or {"key_fields": {"give": "A", "get": "B"}, "give": "A", "get": "B"})
    store.add(p)
    return p


def test_undo_reverts_to_proposed(webapp):
    user = webapp.make_user("owner")
    store = PgProposalStore(webapp.db, user.id)
    p = _seed_proposal(store)
    webapp.auth_as(user)
    webapp.client.post(f"/proposals/{p.id}/approve")
    r = webapp.client.post(f"/proposals/{p.id}/undo").json()
    assert r["status"] == "proposed"
    webapp.db.expire_all()
    assert store.get(p.id).status == ProposalStatus.proposed


def test_confirm_without_league_is_unconfirmed(webapp):
    user = webapp.make_user("owner")
    store = PgProposalStore(webapp.db, user.id)
    p = _seed_proposal(store)  # payload has no "league_id"
    webapp.auth_as(user)
    webapp.client.post(f"/proposals/{p.id}/approve")
    r = webapp.client.post(f"/proposals/{p.id}/confirm").json()
    assert r["confirmed"] is False
    assert "No league" in r["detail"]
    webapp.db.expire_all()
    assert store.get(p.id).status == ProposalStatus.approved  # still approved, not executed


def test_leagues_endpoint_lists_only_my_leagues(webapp):
    user = webapp.make_user("owner")
    lg = add_league(webapp.db, user, espn_league_id=555, team_id=1, season=2025, name="Z")
    webapp.auth_as(user)
    d = webapp.client.get("/api/leagues").json()
    assert [x["league_id"] for x in d["leagues"]] == [str(lg.id)]
    assert d["leagues"][0]["espn_league_id"] == 555 and d["active"] == str(lg.id)
