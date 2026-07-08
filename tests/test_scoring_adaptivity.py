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


# ── Phase 4: returner / IDP / HC scoring + per-N parse ─────────────────────────
def test_return_yards_only_score_when_rule_set():
    """0.25/return-yd applies only when the league scores it."""
    line = {"kickoff_return_yards": 100.0, "punt_return_yards": 40.0}
    no_rule = ScoringEngine(_settings(1.0)).score_statline(line, "WR")
    assert no_rule == 0.0  # nothing in BASE_SCORING pays return yards
    ret_league = _settings(1.0)
    ret_league.scoring["kickoff_return_yards"] = 0.25
    ret_league.scoring["punt_return_yards"] = 0.25
    scored = ScoringEngine(ret_league).score_statline(line, "WR")
    assert scored == 100.0 * 0.25 + 40.0 * 0.25  # 35.0


def test_idp_statline_scores():
    """A defender's solo/assist/sack/PD line scores under IDP rules."""
    idp = LeagueSettings(scoring={
        "def_tackles_solo": 1.0, "def_tackle_assists": 0.5,
        "dst_sacks": 2.0, "def_passes_defended": 1.0,
    })
    line = {"def_tackles_solo": 6.0, "def_tackle_assists": 4.0,
            "dst_sacks": 1.0, "def_passes_defended": 2.0}
    pts = ScoringEngine(idp).score_statline(line, "LB")
    assert pts == 6 * 1.0 + 4 * 0.5 + 1 * 2.0 + 2 * 1.0  # 12.0


def test_hc_expected_points_symmetric():
    """hc_expected_points(0.7)==2.0 for a +5/-5 win/loss league."""
    from fantasy.valuation.hc import hc_expected_points

    league = LeagueSettings(scoring={"hc_team_win": 5.0, "hc_team_loss": -5.0})
    assert hc_expected_points(league, 0.7) == 2.0  # .7*5 + .3*-5
    assert hc_expected_points(league, 0.5) == 0.0
    assert hc_expected_points(league, 1.0) == 5.0


def test_parse_scoring_per_n_normalization():
    """Per-N ESPN variants normalize to per-unit canonical points (Phase 1)."""
    from fantasy.espn.client import EspnClient

    items = [
        {"statId": 58, "points": 0.25},                 # receiving targets (per-unit)
        {"statId": 108, "points": 1.0},                 # solo tackles (per-unit)
        {"statId": 110, "points": 3.0},                 # total tackles per 3 -> 1.0/unit
        {"statId": 114, "points": 0.25},                # KR yards (per-unit)
        {"statId": 116, "points": 2.5},                 # KR yards per 10 -> 0.25/yd
        {"statId": 155, "points": 5.0},                 # HC team win
        {"statId": 156, "points": -5.0},                # HC team loss
    ]
    canonical, raw_map, _ = EspnClient._parse_scoring(items)
    assert canonical["receiving_targets"] == 0.25
    assert canonical["def_tackles_solo"] == 1.0
    assert canonical["def_tackles_total"] == 3.0 / 3  # per-3 -> 1.0/unit
    # KR yards defined twice (per-unit 0.25 + per-10 0.25) -> accumulates.
    assert abs(canonical["kickoff_return_yards"] - (0.25 + 2.5 / 10)) < 1e-9
    assert canonical["hc_team_win"] == 5.0
    assert canonical["hc_team_loss"] == -5.0
    assert raw_map[58] == 0.25


def test_parse_scoring_per_n_is_order_independent():
    """Per-unit (114) + per-N (116) rules must combine additively regardless of
    the order ESPN lists them (a later per-unit item used to overwrite the
    accumulated per-N contribution)."""
    from fantasy.espn.client import EspnClient

    items = [{"statId": 114, "points": 0.1}, {"statId": 116, "points": 2.5}]
    fwd, _, _ = EspnClient._parse_scoring(items)
    rev, _, _ = EspnClient._parse_scoring(list(reversed(items)))
    assert abs(fwd["kickoff_return_yards"] - 0.35) < 1e-9  # 0.1 + 2.5/10
    assert fwd["kickoff_return_yards"] == rev["kickoff_return_yards"]
