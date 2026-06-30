"""Decision-audit core + trade-priority ordering."""

from __future__ import annotations

from fantasy.analysis.audit import RP, diff_team, startsit_audit
from fantasy.league_settings import LeagueSettings, RosterRequirements
from fantasy.notify.base import render_text
from fantasy.orchestrator.cycle import order_for_notification
from fantasy.orchestrator.models import Proposal, ProposalKind

LEAGUE = LeagueSettings(
    team_count=2,
    roster=RosterRequirements(slots={"QB": 1, "RB": 1, "BE": 2}),
)


def _rp(eid, pos):
    return RP(espn_id=eid, name=eid, position=pos, gsis=None)


def test_diff_team_distinguishes_trade_from_waiver():
    # wk1: team1 has {A(rb), C(wr)}, team2 has {B(rb)}.
    # wk2: A<->B swapped (reciprocal => trade); C dropped to nobody (waiver drop).
    rosters = {
        1: {1: {"A": _rp("A", "RB"), "C": _rp("C", "WR")}, 2: {"B": _rp("B", "RB")}},
        2: {1: {"B": _rp("B", "RB")}, 2: {"A": _rp("A", "RB")}},
    }
    transitions, trades = diff_team(rosters, 1)
    assert len(trades) == 1
    t = trades[0]
    assert [p.espn_id for p in t.received] == ["B"]
    assert [p.espn_id for p in t.sent] == ["A"]
    # C left to nobody -> a waiver/drop transition, NOT a trade.
    dropped = [p.espn_id for tr in transitions for p in tr.dropped]
    assert "C" in dropped
    assert all("C" not in [p.espn_id for p in tr.received] for tr in [t])


def test_startsit_left_on_bench_and_biggest_miss():
    roster = {"qb": _rp("qb", "QB"), "rb1": _rp("rb1", "RB"), "rb2": _rp("rb2", "RB")}
    rosters = {1: {7: roster}}
    box = {"qb": {1: 20.0}, "rb1": {1: 5.0}, "rb2": {1: 18.0}}
    started = {"qb": {1: 20.0}, "rb1": {1: 5.0}}  # rb2 benched
    out = startsit_audit(rosters, box, started, 7, LEAGUE, [1])
    wk = out["weeks"][0]
    assert wk["started"] == 25.0          # qb 20 + rb1 5
    assert wk["optimal"] == 38.0          # qb 20 + best rb (rb2 18)
    assert wk["left"] == 13.0
    assert out["total_left_on_bench"] == 13.0
    assert wk["biggest"]["bench"] == "rb2" and wk["biggest"]["over"] == "rb1"
    assert wk["biggest"]["gain"] == 13.0


def _prop(kind, value):
    return Proposal(kind=kind, season=2025, week=5, title=f"{kind.value} {value}", value=value)


def test_trades_float_first_and_get_tagged():
    gen = [_prop(ProposalKind.waiver, 99.0),       # higher value than the trade
           _prop(ProposalKind.start_sit, 50.0),
           _prop(ProposalKind.trade, 3.0)]
    ordered = order_for_notification(gen)
    assert ordered[0].kind == ProposalKind.trade   # trade first despite lower value
    assert ordered[0].payload.get("priority") is True
    # non-trades keep value-descending order after the trade
    assert [p.kind for p in ordered[1:]] == [ProposalKind.waiver, ProposalKind.start_sit]


def test_render_text_stars_priority_trade():
    p = _prop(ProposalKind.trade, 5.0)
    order_for_notification([p])                      # tags priority
    assert "PRIORITY" in render_text(p)
    assert "PRIORITY" not in render_text(_prop(ProposalKind.waiver, 5.0))
