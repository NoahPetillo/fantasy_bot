"""Backup-spike usage redistribution + player-prop conversion math."""

from __future__ import annotations

import pandas as pd

from fantasy.league_settings import LeagueSettings
from fantasy.projections.props import american_to_prob, devig_two_way, statline_to_points
from fantasy.projections.usage import vacated_boosts
from fantasy.valuation.scoring import ScoringEngine

_BOARD = pd.DataFrame([
    {"player_id": "rb1", "team": "SF", "position": "RB", "proj": 16.0},
    {"player_id": "rb2", "team": "SF", "position": "RB", "proj": 6.0},
    {"player_id": "rb3", "team": "SF", "position": "RB", "proj": 3.0},
    {"player_id": "wr1", "team": "SF", "position": "WR", "proj": 14.0},
    {"player_id": "rb_other", "team": "KC", "position": "RB", "proj": 9.0},
])


def test_backup_spikes_when_starter_out():
    boosts = vacated_boosts(["rb1"], _BOARD)
    assert "rb2" in boosts                          # highest-proj other SF RB gets it
    assert "rb3" not in boosts and "rb_other" not in boosts
    assert abs(boosts["rb2"] - 0.55 * 16.0) < 0.01  # RB redistribution share
    assert boosts["rb2"] <= 16.0                    # never exceeds vacated value


def test_no_same_team_backup_no_boost():
    one = pd.DataFrame([{"player_id": "wr1", "team": "SF", "position": "WR", "proj": 12.0}])
    assert vacated_boosts(["wr1"], one) == {}


def test_american_odds_and_devig():
    assert abs(american_to_prob(100) - 0.5) < 1e-9
    assert abs(american_to_prob(-200) - (200 / 300)) < 1e-6
    assert abs(devig_two_way(-110, -110) - 0.5) < 1e-6  # symmetric -> 50/50


def test_prop_statline_to_points():
    ls = LeagueSettings(scoring={"receiving_yards": 0.1, "receptions": 1.0, "receiving_tds": 6.0})
    eng = ScoringEngine(ls)
    pts = statline_to_points({"receiving_yards": 70.0, "receptions": 6.0}, 0.4, eng, "WR")
    # base 7 + 6 = 13; TD bonus = -ln(0.6)*6 ≈ 3.06  -> ~16.1
    assert 15.5 < pts < 16.6
