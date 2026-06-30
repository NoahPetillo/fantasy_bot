"""Phase B: expert signals wired into the decision engine (gated + capped)."""

from __future__ import annotations

import pandas as pd
import pytest

from fantasy.config import settings
from fantasy.decisions.waivers import recommend_waivers
from fantasy.league_settings import LeagueSettings, RosterRequirements
from fantasy.league_state import LeagueSnapshot
from fantasy.news.experts.models import FusedSignal
from fantasy.news.models import EventType
from fantasy.orchestrator.cycle import _expert_alerts
from fantasy.projections.service import ProjectionService

LEAGUE = LeagueSettings(
    team_count=2, scoring={"receptions": 1.0, "receiving_yards": 0.1},
    roster=RosterRequirements(slots={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "BE": 2}),
)


@pytest.fixture
def adjust_on():
    prev = settings.expert_adjust_decisions
    settings.expert_adjust_decisions = True
    yield
    settings.expert_adjust_decisions = prev


def _fused(pid, direction, etype=EventType.injury_out, corro=True, conf=0.9, trust=0.95):
    return FusedSignal(player_id=pid, player_name="X", event_type=etype, direction=direction,
                       fused_confidence=conf, trust_weight=trust, independent_outlets=2,
                       corroborated=corro)


def test_projection_delta_capped_and_gated(adjust_on):
    svc = ProjectionService(LEAGUE)
    cur = pd.DataFrame({"player_id": ["p1", "p2"], "proj": [100.0, 50.0]})
    fused = [_fused("p1", -1)]  # corroborated injury_out

    out = svc._apply_expert_deltas(cur.copy(), fused)
    p1 = out.loc[out.player_id == "p1", "proj"].iloc[0]
    assert p1 < 100.0 and p1 >= 85.0  # reduced, capped at -15%
    assert out.loc[out.player_id == "p2", "proj"].iloc[0] == 50.0  # untouched

    settings.expert_adjust_decisions = False
    off = svc._apply_expert_deltas(cur.copy(), fused)
    assert off.loc[off.player_id == "p1", "proj"].iloc[0] == 100.0  # flag gates it


def test_uncorroborated_signal_does_not_move_projection(adjust_on):
    svc = ProjectionService(LEAGUE)
    cur = pd.DataFrame({"player_id": ["p1"], "proj": [100.0]})
    out = svc._apply_expert_deltas(cur.copy(), [_fused("p1", -1, corro=False)])
    assert out.loc[0, "proj"] == 100.0  # gate: only corroborated signals adjust


def _waiver_setup():
    rows = [("s_qb", "QB", 20, 6), ("s_rb1", "RB", 18, 9), ("s_rb2", "RB", 16, 7),
            ("s_wr1", "WR", 17, 8), ("s_wr2", "WR", 15, 6), ("s_te", "TE", 12, 4),
            ("s_flex", "RB", 13, 5), ("bench", "WR", 6, -4), ("fa_good", "WR", 11, 3)]
    board = pd.DataFrame([{"player_id": p, "player_display_name": p.upper(), "position": pos,
                           "proj": pr, "vor": v} for p, pos, pr, v in rows])
    snap = LeagueSnapshot(season=2024, week=6, my_team_id=1,
                          teams={1: ["s_qb", "s_rb1", "s_rb2", "s_wr1", "s_wr2", "s_te", "s_flex", "bench"]},
                          free_agents=["fa_good"], names={}, positions={}, faab_remaining={1: 100})
    return board, snap


def test_waiver_boost_raises_value(adjust_on):
    board, snap = _waiver_setup()
    base = recommend_waivers(snap, board, LEAGUE, remaining_weeks=10)
    boosted = recommend_waivers(snap, board, LEAGUE, remaining_weeks=10, boosts={"fa_good": 1.5})
    assert base and boosted
    assert boosted[0].value > base[0].value           # expert boost raised waiver value
    assert boosted[0].payload["expert_boost"] == 1.5


def test_expert_alert_for_my_injured_player():
    snap = LeagueSnapshot(season=2024, week=6, my_team_id=1, teams={1: ["mine"]},
                          free_agents=["fa1"], names={}, positions={})
    fused = [FusedSignal(player_id="mine", player_name="My Guy", event_type=EventType.injury_out,
                         direction=-1, fused_confidence=0.9, trust_weight=0.95,
                         independent_outlets=2, corroborated=True, experts=["@AdamSchefter"])]
    alerts = _expert_alerts(snap, fused)
    assert alerts and "My Guy" in alerts[0].title and alerts[0].kind.value == "alert"
