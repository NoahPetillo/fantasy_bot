"""Phase 2 decision logic: lineup LP, FAAB sizing, store idempotency, trade guards."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from fantasy.decisions.faab import suggest_bid
from fantasy.decisions.lineup import lineup_value, optimize_lineup
from fantasy.decisions.trades import evaluate_trade_package, recommend_trades
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


# ── manual N-for-M trade analyzer (evaluate_trade_package) ────────────────────
LEAGUE_SF = LeagueSettings(  # superflex variant: an OP slot accepts a QB
    team_count=2,
    roster=RosterRequirements(slots={"QB": 1, "OP": 1, "RB": 2, "WR": 2, "TE": 1, "BE": 4}),
)


def _maps(rows):
    """rows: [(pid, pos, ros_proj, ros_vor)] -> (ros, ros_vor, pos, names)."""
    return ({r[0]: r[2] for r in rows}, {r[0]: r[3] for r in rows},
            {r[0]: r[1] for r in rows}, {r[0]: r[0].upper() for r in rows})


def test_package_big_hitter_beats_two_bench_bodies():
    """The core requirement: two players who bring MORE raw points can be worth
    LESS than one stud when only one of them cracks the starting lineup."""
    roster = [
        ("qb1", "QB", 22, 6), ("rb1", "RB", 20, 9), ("rb2", "RB", 18, 7),
        ("wr1", "WR", 17, 8), ("wr2", "WR", 16, 7), ("te1", "TE", 12, 4),
        ("wr3", "WR", 12, 3), ("wrb", "WR", 5, -3),  # wrb is bench filler
    ]
    extra = [("nrb_a", "RB", 16, 6), ("nrb_b", "RB", 15, 5), ("elite", "RB", 25, 14)]
    ros, ros_vor, pos, names = _maps(roster + extra)
    my = [r[0] for r in roster]
    pkg = evaluate_trade_package(my, None, ["wrb"], ["nrb_a", "nrb_b"], ros, ros_vor, pos,
                                 LEAGUE, bench_size=4, single_counterparty=False, names=names)
    solo = evaluate_trade_package(my, None, ["wrb"], ["elite"], ros, ros_vor, pos,
                                  LEAGUE, bench_size=4, single_counterparty=False, names=names)
    assert pkg["points_sum_delta"] > solo["points_sum_delta"]   # package brings more raw points
    assert solo["lineup_delta"] > pkg["lineup_delta"]           # but the stud helps the lineup more
    assert pkg["lineup_delta"] < pkg["points_sum_delta"]        # bench glut: sum overstates the gain
    assert any(g["id"] == "nrb_b" and not g["starter"] for g in pkg["get"])  # 2nd RB rides the bench


def test_package_superflex_second_qb_fills_op():
    roster = [("qb1", "QB", 24, 10), ("rb1", "RB", 18, 7), ("rb2", "RB", 16, 6),
              ("wr1", "WR", 17, 7), ("wr2", "WR", 15, 6), ("te1", "TE", 11, 4), ("wrb", "WR", 4, -4)]
    ros, ros_vor, pos, names = _maps(roster + [("qb2", "QB", 21, 8)])
    my = [r[0] for r in roster]
    r = evaluate_trade_package(my, None, ["wrb"], ["qb2"], ros, ros_vor, pos, LEAGUE_SF,
                               bench_size=4, single_counterparty=False, names=names)
    assert r["lineup_delta"] > 0
    assert any(g["id"] == "qb2" and g["starter"] for g in r["get"])  # QB fills the superflex slot


def test_package_trading_depth_has_a_cost():
    roster = [("qb1", "QB", 22, 6), ("rb1", "RB", 20, 9), ("rb2", "RB", 18, 7),
              ("wr1", "WR", 17, 8), ("wr2", "WR", 16, 7), ("te1", "TE", 12, 4),
              ("wr3", "WR", 12, 3), ("benchrb", "RB", 3, 5)]  # low proj, positive value: depth
    ros, ros_vor, pos, names = _maps(roster)
    my = [r[0] for r in roster]
    r = evaluate_trade_package(my, None, ["benchrb"], ["kx"], ros, ros_vor, pos, LEAGUE,
                               bench_size=4, single_counterparty=False, names={**names, "kx": "KX"})
    assert r["depth_delta"] < 0                                   # shipping useful depth costs something
    assert any("No projection" in n for n in r["notes"])          # kx has no projection


def test_package_over_capacity_flags_drops():
    roster = [("qb1", "QB", 22, 6), ("rb1", "RB", 20, 9), ("rb2", "RB", 18, 7),
              ("wr1", "WR", 17, 8), ("wr2", "WR", 16, 7), ("te1", "TE", 12, 4),
              ("wr3", "WR", 12, 3), ("b1", "WR", 6, 0), ("b2", "RB", 6, 0),
              ("b3", "TE", 6, 0), ("b4", "QB", 6, 0)]  # 11 = starters(7) + bench(4)
    extra = [("g1", "RB", 5, 0), ("g2", "WR", 5, 0), ("g3", "WR", 5, 0)]
    ros, ros_vor, pos, names = _maps(roster + extra)
    my = [r[0] for r in roster]
    r = evaluate_trade_package(my, None, ["b1"], ["g1", "g2", "g3"], ros, ros_vor, pos, LEAGUE,
                               bench_size=4, single_counterparty=False, names=names)
    assert r["need_to_drop"] == 2                                 # 11 - 1 + 3 = 13, cap 11
    assert any("roster limit" in n for n in r["notes"])


def test_package_ir_slots_dont_count_as_over_limit():
    """ESPN rosters include IR players, so the capacity must include IR too —
    otherwise every trade in an IR league would falsely warn 'drop players'."""
    roster = [("qb1", "QB", 22, 6), ("rb1", "RB", 20, 9), ("rb2", "RB", 18, 7),
              ("wr1", "WR", 17, 8), ("wr2", "WR", 16, 7), ("te1", "TE", 12, 4),
              ("wr3", "WR", 12, 3),                                # 7 starters
              ("b1", "RB", 6, 0), ("b2", "WR", 6, 0), ("b3", "TE", 6, 0), ("b4", "QB", 6, 0),  # 4 bench
              ("ir1", "WR", 0, 0), ("ir2", "RB", 0, 0)]           # 2 injured, parked on IR
    ros, ros_vor, pos, names = _maps(roster)
    my = [r[0] for r in roster]                                   # 13 rostered incl. IR
    r = evaluate_trade_package(my, None, ["b1"], ["nrb"], {**ros, "nrb": 15}, {**ros_vor, "nrb": 5},
                               {**pos, "nrb": "RB"}, LEAGUE, bench_size=4, ir_size=2,
                               single_counterparty=False, names={**names, "nrb": "NRB"})
    assert r["need_to_drop"] == 0                                 # 13 - 1 + 1 = 13, cap 7+4+2
    without_ir = evaluate_trade_package(my, None, ["b1"], ["nrb"], {**ros, "nrb": 15},
                                        {**ros_vor, "nrb": 5}, {**pos, "nrb": "RB"}, LEAGUE,
                                        bench_size=4, ir_size=0, single_counterparty=False,
                                        names={**names, "nrb": "NRB"})
    assert without_ir["need_to_drop"] == 2                        # proves IR capacity is what fixes it


def test_package_single_acquisition_that_misses_lineup_gets_a_note():
    """A 1-for-1 where the acquired player doesn't crack the lineup is exactly the
    case the feature explains — the bench note must fire even for a single player."""
    roster = [("qb1", "QB", 22, 6), ("rb1", "RB", 20, 9), ("rb2", "RB", 18, 7),
              ("wr1", "WR", 17, 8), ("wr2", "WR", 16, 7), ("te1", "TE", 12, 4),
              ("wr3", "WR", 12, 3), ("wrb", "WR", 5, -3)]
    ros, ros_vor, pos, names = _maps(roster + [("nwr", "WR", 8, -1)])  # below the FLEX starter
    my = [r[0] for r in roster]
    r = evaluate_trade_package(my, None, ["wrb"], ["nwr"], ros, ros_vor, pos, LEAGUE,
                               bench_size=4, single_counterparty=False, names=names)
    assert not r["get"][0]["starter"]
    assert any("crack your starting lineup" in n for n in r["notes"])


def test_package_verdict_reflects_depth_loss():
    """Shipping meaningful bench insurance for no lineup change must NOT read 'neutral';
    the verdict keys off the depth-adjusted delta."""
    roster = [("qb1", "QB", 22, 6), ("rb1", "RB", 20, 9), ("rb2", "RB", 18, 7),
              ("wr1", "WR", 17, 8), ("wr2", "WR", 16, 7), ("te1", "TE", 12, 4),
              ("wr3", "WR", 12, 3), ("stash", "RB", 3, 16)]  # valuable body that isn't starting
    ros, ros_vor, pos, names = _maps(roster)
    my = [r[0] for r in roster]
    r = evaluate_trade_package(my, None, ["stash"], ["junk"], ros, ros_vor, pos, LEAGUE,
                               bench_size=4, single_counterparty=False, names={**names, "junk": "JUNK"})
    assert r["lineup_delta"] == 0.0 and r["depth_delta"] <= -2.0
    assert r["verdict"] == "unfavorable"


def test_package_multi_team_has_no_accept_prob():
    ros, ros_vor, pos, names = _maps([("a", "RB", 10, 2), ("b", "WR", 10, 2)])
    r = evaluate_trade_package(["a"], None, ["a"], ["b"], ros, ros_vor, pos, LEAGUE,
                               bench_size=4, single_counterparty=False, names=names)
    assert r["accept_prob"] is None


def test_package_single_counterparty_has_accept_prob():
    roster = [("qb1", "QB", 22, 6), ("rb1", "RB", 20, 9), ("rb2", "RB", 18, 7),
              ("wr1", "WR", 17, 8), ("wr2", "WR", 16, 7), ("te1", "TE", 12, 4), ("wr3", "WR", 12, 3)]
    opp = [("orb", "RB", 21, 10), ("owr", "WR", 8, -1), ("owr2", "WR", 9, 0), ("oqb", "QB", 19, 4),
           ("ote", "TE", 9, 2), ("orb2", "RB", 7, -2), ("owr3", "WR", 7, -2)]
    ros, ros_vor, pos, names = _maps(roster + opp)
    r = evaluate_trade_package([x[0] for x in roster], [x[0] for x in opp], ["wr3"], ["orb"],
                               ros, ros_vor, pos, LEAGUE, bench_size=4, single_counterparty=True, names=names)
    assert r["accept_prob"] is not None and 0.0 <= r["accept_prob"] <= 1.0


def test_package_empty_roster_is_safe():
    ros, ros_vor, pos, names = _maps([("x", "RB", 10, 2)])
    r = evaluate_trade_package([], None, [], ["x"], ros, ros_vor, pos, LEAGUE,
                               bench_size=4, single_counterparty=False, names=names)
    assert r["lineup_before"] == 0.0 and r["lineup_after"] == 10.0


def test_trade_block_covers_all_rostered_players():
    from fantasy.api.dashboard_data import _trade_block

    board = _synth_board()
    me, opp = [f"p{i}" for i in range(8)], [f"p{i}" for i in range(8, 16)]
    snap = LeagueSnapshot(
        season=2024, week=5, my_team_id=1, teams={1: me, 2: opp}, free_agents=[],
        names={r.player_id: r.player_display_name for r in board.itertuples()},
        positions={r.player_id: r.position for r in board.itertuples()},
        team_names={1: "MY TEAM", 2: "Team 2"},
    )
    # A rostered player with no board row (e.g. a kicker) must still appear, priced at 0.
    snap.teams[1].append("kx")
    snap.names["kx"] = "Kicker X"
    snap.positions["kx"] = "K"
    tb = _trade_block(snap, board.set_index("player_id"), LEAGUE, 8)
    assert set(tb["players"]) == set(me + opp + ["kx"])
    assert tb["players"]["p0"]["team_id"] == 1 and tb["players"]["p8"]["team_id"] == 2
    assert tb["players"]["kx"] == {"name": "Kicker X", "pos": "K", "proj": 0.0, "vor": 0.0,
                                   "team_id": 1, "games": 8}
    assert tb["remaining_weeks"] == 8 and tb["bench_size"] == 4
    assert tb["roster_slots"] == {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1}
