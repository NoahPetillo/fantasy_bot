"""Phase 2 decision logic: lineup LP, FAAB sizing, store idempotency, trade guards."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from fantasy.decisions.faab import suggest_bid
from fantasy.decisions.lineup import lineup_value, optimize_lineup
from fantasy.decisions.trades import recommend_trades
from fantasy.decisions.waivers import recommend_waivers
from fantasy.league_settings import LeagueSettings, RosterRequirements
from fantasy.league_state import LeagueSnapshot
from fantasy.orchestrator.models import Proposal, ProposalKind, ProposalStatus
from fantasy.orchestrator.store import Store

LEAGUE = LeagueSettings(
    team_count=2,
    roster=RosterRequirements(slots={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "BE": 4}),
)


def test_lineup_optimizer_respects_slots_and_maximizes():
    players = [
        ("qb1", "QB", 25), ("qb2", "QB", 10),
        ("rb1", "RB", 20), ("rb2", "RB", 15), ("rb3", "RB", 12),
        ("wr1", "WR", 18), ("wr2", "WR", 14), ("wr3", "WR", 9),
        ("te1", "TE", 11),
    ]
    lineup = optimize_lineup(players, LEAGUE)
    starters = {pid for pids in lineup.values() for pid in pids}
    # QB is unambiguous; the full starter set is invariant even though which RB
    # lands in RB vs FLEX is a tie-broken choice.
    assert lineup["QB"] == ["qb1"]
    assert starters == {"qb1", "rb1", "rb2", "rb3", "wr1", "wr2", "te1"}
    assert "wr3" not in starters and "qb2" not in starters
    proj = {p: v for p, _, v in players}
    assert sum(proj[s] for s in starters) == 115


def test_lineup_value_matches_optimizer_total():
    proj = {"qb1": 25, "rb1": 20, "rb2": 15, "rb3": 12, "wr1": 18, "wr2": 14, "te1": 11}
    pos = {"qb1": "QB", "rb1": "RB", "rb2": "RB", "rb3": "RB",
           "wr1": "WR", "wr2": "WR", "te1": "TE"}
    val = lineup_value(proj, pos, list(proj), LEAGUE)
    assert val == 25 + 20 + 15 + 18 + 14 + 11 + 12  # FLEX=rb3


def test_faab_bid_is_monotonic_and_bounded():
    assert suggest_bid(0, 100) == 0
    low, high = suggest_bid(10, 100), suggest_bid(80, 100)
    assert 0 < low < high <= 100
    assert suggest_bid(500, 100, remaining_budget=15) <= 15  # capped at remaining


def test_store_idempotency_and_executed_guard():
    db = Path(tempfile.mkdtemp()) / "t.sqlite"
    store = Store(db)
    p1 = Proposal(kind=ProposalKind.waiver, season=2024, week=5, team_id=1,
                  title="Add X", payload={"key_fields": {"add": "X", "drop": "Y"}})
    p2 = Proposal(kind=ProposalKind.waiver, season=2024, week=5, team_id=1,
                  title="Add X (dup)", payload={"key_fields": {"add": "X", "drop": "Y"}})
    assert p1.idempotency_key == p2.idempotency_key
    assert store.add(p1) is True
    assert store.add(p2) is False  # duplicate suppressed
    assert not store.has_executed(p1.idempotency_key)
    store.set_status(p1.id, ProposalStatus.executed)
    assert store.has_executed(p1.idempotency_key)


def _synth_board():
    # 2-team league worth of players; my surplus is RB, opp surplus is WR.
    rows = []
    for i, (pos, proj, vor) in enumerate([
        ("QB", 22, 6), ("RB", 18, 9), ("RB", 16, 7), ("RB", 14, 5), ("RB", 13, 4),
        ("WR", 17, 8), ("WR", 9, 1), ("TE", 11, 4),
        ("QB", 21, 5), ("WR", 18, 9), ("WR", 16, 7), ("WR", 15, 6), ("WR", 10, 2),
        ("RB", 9, 1), ("RB", 8, 0), ("TE", 10, 3),
    ]):
        rows.append({"player_id": f"p{i}", "player_display_name": f"P{i}",
                     "position": pos, "proj": proj, "vor": vor})
    return pd.DataFrame(rows)


def _waiver_board(bench_vor, fa_vor):
    """My 7 starters + 1 weak bench player; plus two free agents."""
    rows = [
        ("s_qb", "QB", 20, 6), ("s_rb1", "RB", 18, 9), ("s_rb2", "RB", 16, 7),
        ("s_wr1", "WR", 17, 8), ("s_wr2", "WR", 15, 6), ("s_te", "TE", 12, 4),
        ("s_flex", "RB", 13, 5), ("bench", "WR", 6, bench_vor),
        ("fa_good", "WR", 11, fa_vor), ("fa_repl", "WR", 7, -2),
    ]
    return pd.DataFrame([
        {"player_id": pid, "player_display_name": pid.upper(), "position": pos,
         "proj": proj, "vor": vor} for pid, pos, proj, vor in rows
    ])


def _waiver_snap(roster, fas):
    return LeagueSnapshot(
        season=2024, week=6, my_team_id=1, teams={1: roster}, free_agents=fas,
        names={}, positions={}, faab_remaining={1: 100},
    )


def test_waiver_upgrades_bench_with_named_drop():
    board = _waiver_board(bench_vor=-4, fa_vor=3)  # fa_good clearly beats the weak bench WR
    snap = _waiver_snap(
        ["s_qb", "s_rb1", "s_rb2", "s_wr1", "s_wr2", "s_te", "s_flex", "bench"],
        ["fa_good", "fa_repl"],
    )
    props = recommend_waivers(snap, board, LEAGUE, remaining_weeks=10)
    assert props, "expected a bench-depth waiver upgrade"
    top = props[0]
    assert top.payload["add"] == "fa_good"      # the better player
    assert top.payload["drop"] == "bench"       # drops the weakest bench player, not a starter
    assert top.value > 0


def test_no_waiver_when_pool_is_replacement_level():
    board = _waiver_board(bench_vor=2, fa_vor=-1)  # bench is fine; FAs are worse
    snap = _waiver_snap(
        ["s_qb", "s_rb1", "s_rb2", "s_wr1", "s_wr2", "s_te", "s_flex", "bench"],
        ["fa_good", "fa_repl"],
    )
    assert recommend_waivers(snap, board, LEAGUE, remaining_weeks=10) == []


def test_trades_only_propose_win_wins():
    board = _synth_board()
    me = [f"p{i}" for i in range(8)]       # RB-heavy
    opp = [f"p{i}" for i in range(8, 16)]  # WR-heavy
    snap = LeagueSnapshot(
        season=2024, week=5, my_team_id=1, teams={1: me, 2: opp}, free_agents=[],
        names={r.player_id: r.player_display_name for r in board.itertuples()},
        positions={r.player_id: r.position for r in board.itertuples()},
        team_names={1: "MY TEAM", 2: "Team 2"},
    )
    props = recommend_trades(snap, board, LEAGUE, remaining_weeks=8)
    for p in props:
        assert p.value > 0                # my lineup improves
        assert 0.2 <= p.confidence <= 1.0  # opponent plausibly accepts
        assert p.payload["give"] in me and p.payload["get"] in opp
