"""The league-adaptivity guarantee, as executable tests.

The SAME stat line must score differently under different league rules, and the
roster math (replacement baselines) must shift with roster slots — proving the
system reads from LeagueSettings rather than hardcoding a format.
"""

from __future__ import annotations

import pandas as pd

from fantasy.league_settings import LeagueSettings, RosterRequirements
from fantasy.valuation.scoring import ScoringEngine

# A WR stat line: 6 catches, 90 rec yards, 1 TD, 10 rush yards.
WR_LINE = {
    "receptions": 6.0,
    "receiving_yards": 90.0,
    "receiving_tds": 1.0,
    "rushing_yards": 10.0,
}

BASE_SCORING = {
    "passing_yards": 0.04,
    "passing_tds": 4.0,
    "passing_interceptions": -2.0,
    "rushing_yards": 0.1,
    "rushing_tds": 6.0,
    "receiving_yards": 0.1,
    "receiving_tds": 6.0,
    "fumbles_lost": -2.0,
}


def _settings(rec_points: float, te_bonus: float | None = None, **roster) -> LeagueSettings:
    scoring = dict(BASE_SCORING)
    scoring["receptions"] = rec_points
    ls = LeagueSettings(scoring=scoring)
    if te_bonus is not None:
        ls.position_reception_bonus = {"TE": te_bonus}
    if roster:
        ls.roster = RosterRequirements(slots=roster)
    return ls


def test_ppr_format_changes_score():
    # Base points (no receptions): 90*.1 + 6(TD) + 10*.1 = 9 + 6 + 1 = 16.0
    standard = ScoringEngine(_settings(0.0)).score_statline(WR_LINE, "WR")
    half = ScoringEngine(_settings(0.5)).score_statline(WR_LINE, "WR")
    ppr = ScoringEngine(_settings(1.0)).score_statline(WR_LINE, "WR")
    assert standard == 16.0
    assert half == 16.0 + 6 * 0.5  # +3
    assert ppr == 16.0 + 6 * 1.0  # +6
    assert standard < half < ppr


def test_te_premium_only_helps_tes():
    eng = ScoringEngine(_settings(0.5, te_bonus=0.5))
    wr = eng.score_statline(WR_LINE, "WR")
    te = eng.score_statline(WR_LINE, "TE")
    assert te == wr + 6 * 0.5  # TE gets the extra 0.5/rec, WR does not


def test_six_point_passing_td_is_respected():
    qb_line = {"passing_yards": 300.0, "passing_tds": 3.0, "passing_interceptions": 1.0}
    four_pt = _settings(0.0)
    six_pt = _settings(0.0)
    six_pt.scoring["passing_tds"] = 6.0
    s4 = ScoringEngine(four_pt).score_statline(qb_line, "QB")
    s6 = ScoringEngine(six_pt).score_statline(qb_line, "QB")
    assert s6 == s4 + 3 * 2.0  # +2 per TD * 3 TDs


def test_superflex_changes_qb_replacement_demand():
    one_qb = _settings(1.0, QB=1, RB=2, WR=2, TE=1, FLEX=1)
    superflex = _settings(1.0, QB=1, RB=2, WR=2, TE=1, FLEX=1, OP=1)
    assert not one_qb.roster.has_superflex
    assert superflex.roster.has_superflex
    # Superflex demands materially more QB starters league-wide.
    assert superflex.roster.starters_at_position("QB") > one_qb.roster.starters_at_position("QB")
    assert superflex.num_qbs_effective == 2


def test_flex_adds_fractional_position_demand():
    ls = _settings(1.0, QB=1, RB=2, WR=2, TE=1, FLEX=1)
    # RB demand = 2 dedicated + 1/3 of the FLEX.
    assert abs(ls.roster.starters_at_position("RB") - (2 + 1 / 3)) < 1e-9


def test_vectorized_matches_scalar():
    eng = ScoringEngine(_settings(1.0))
    df = pd.DataFrame(
        [
            {"position": "WR", "receptions": 6, "receiving_yards": 90,
             "receiving_tds": 1, "rushing_yards": 10},
        ]
    )
    vec = eng.score_dataframe(df).iloc[0]
    scalar = eng.score_statline(WR_LINE, "WR")
    assert abs(vec - scalar) < 1e-9
