"""Season draft-value board — assembled offline from synthetic source frames.

Monkeypatches the four external sources (ESPN kona, Sleeper season, FFC ADP, the
returner overlay) so the pipeline math is tested with no network: per-stat
scoring under the league's rules, the return overlay adding points, IDP + HC rows
appearing with the right positions, VOR computed, and the prior-season fallback
when both projection sources are empty.
"""

from __future__ import annotations

import pandas as pd
import pytest

import fantasy.draft.season_board as sb
from fantasy.league_settings import LeagueSettings, RosterRequirements

# The custom league: 0.25/target, 0.25/return-yd, IDP (DP slot), HC (+5/-5).
_SCORING = {
    "passing_yards": 0.04, "passing_tds": 4.0,
    "rushing_yards": 0.1, "rushing_tds": 6.0,
    "receiving_yards": 0.1, "receiving_tds": 6.0,
    "receptions": 0.5, "receiving_targets": 0.25,
    "kickoff_return_yards": 0.25, "punt_return_yards": 0.25,
    "def_tackles_solo": 1.0, "def_tackle_assists": 0.5, "dst_sacks": 2.0,
    "hc_team_win": 5.0, "hc_team_loss": -5.0,
}
_ROSTER = RosterRequirements(
    slots={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 2, "K": 1, "DP": 1, "HC": 1, "BE": 6}
)


def _league() -> LeagueSettings:
    return LeagueSettings(league_id=1, season=2026, team_count=12,
                          scoring=dict(_SCORING), roster=_ROSTER)


@pytest.fixture
def patched_sources(monkeypatch):
    """Wire synthetic ESPN/Sleeper/FFC/overlay frames into the board pipeline."""
    espn = pd.DataFrame([
        # A target-heavy WR: 100 tgt, 80 rec, 1200 yds, 8 TD.
        {"player_id": "WR1", "name": "Alpha WR", "position": "WR", "team": "AAA",
         "receiving_targets": 100.0, "receptions": 80.0, "receiving_yards": 1200.0,
         "receiving_tds": 8.0},
        {"player_id": "RB1", "name": "Bell Cow", "position": "RB", "team": "BBB",
         "rushing_yards": 1400.0, "rushing_tds": 12.0, "receptions": 40.0,
         "receiving_targets": 50.0, "receiving_yards": 300.0},
    ])
    sleeper = pd.DataFrame([
        # Same WR from Sleeper (no targets) — averaged with ESPN's yards.
        {"player_id": "WR1", "name": "Alpha WR", "position": "WR", "team": "AAA",
         "receptions": 80.0, "receiving_yards": 1000.0, "receiving_tds": 8.0,
         "adp_ppr": 5.0},
        # An IDP linebacker.
        {"player_id": "LB1", "name": "Mike Backer", "position": "LB", "team": "CCC",
         "def_tackles_solo": 90.0, "def_tackle_assists": 40.0, "dst_sacks": 3.0,
         "adp_idp": 40.0},
    ])
    ffc = pd.DataFrame([
        {"player_id": "WR1", "name": "Alpha WR", "position": "WR", "team": "AAA",
         "adp": 5.0, "sd": 2.0, "high": 1, "low": 12, "times_drafted": 100},
        {"player_id": "RB1", "name": "Bell Cow", "position": "RB", "team": "BBB",
         "adp": 3.0, "sd": 1.5, "high": 1, "low": 8, "times_drafted": 100},
    ])

    monkeypatch.setattr(sb, "_safe_espn", lambda season, league, refresh: espn.copy())
    monkeypatch.setattr(sb, "_safe_sleeper", lambda season, refresh: sleeper.copy())
    monkeypatch.setattr(sb, "load_ffc_adp",
                        lambda season, teams=12, fmt="ppr": ffc.copy())
    # Return overlay: WR1 gets 200 season return pts.
    monkeypatch.setattr(sb, "return_points_overlay",
                        lambda league, season, **kw: {"WR1": 200.0})
    # Deterministic 2-team HC EV.
    monkeypatch.setattr(sb, "hc_draft_ev", lambda league, season: pd.DataFrame([
        {"player_id": "HC:AAA", "team": "AAA", "coach_label": "HC AAA",
         "position": "HC", "win_prob": 0.65, "expected_season_points": 25.5},
        {"player_id": "HC:BBB", "team": "BBB", "coach_label": "HC BBB",
         "position": "HC", "win_prob": 0.45, "expected_season_points": -8.5},
    ]))
    return espn, sleeper


def test_board_builds_with_expected_columns(patched_sources):
    board = sb.build_season_board(2026, _league())
    for col in ("player_id", "name", "position", "team", "proj", "return_pts",
                "adp", "adp_sd", "vor", "replacement", "proj_source"):
        assert col in board.columns
    assert board["proj_source"].iloc[0] == "consensus"
    assert len(board) >= 4  # WR1, RB1, LB1, 2 HC


def test_targets_contribute_quarter_point_each(patched_sources):
    board = sb.build_season_board(2026, _league())
    wr = board[board["player_id"] == "WR1"].iloc[0]
    # ESPN-only targets: 100 * 0.25 = 25 pts, always present.
    # yards averaged: (1200+1000)/2 = 1100 -> 110 pts; rec 80*0.5=40; TD 8*6=48.
    # + return overlay 200. Sum = 25 + 110 + 40 + 48 + 200 = 423.
    assert wr["return_pts"] == 200.0
    assert abs(wr["proj"] - (25 + 110 + 40 + 48 + 200)) < 1e-6


def test_return_overlay_lifts_projection(patched_sources):
    board = sb.build_season_board(2026, _league())
    wr = board[board["player_id"] == "WR1"].iloc[0]
    # proj includes the 200 return pts; without them it would be 223.
    assert wr["proj"] - wr["return_pts"] == pytest.approx(223.0)


def test_idp_and_hc_rows_present_with_positions(patched_sources):
    board = sb.build_season_board(2026, _league())
    lb = board[board["player_id"] == "LB1"]
    assert len(lb) == 1 and lb.iloc[0]["position"] == "LB"
    # Solo 90*1 + assist 40*0.5 + sack 3*2 = 90 + 20 + 6 = 116.
    assert abs(lb.iloc[0]["proj"] - 116.0) < 1e-6
    hc = board[board["position"] == "HC"]
    assert set(hc["player_id"]) == {"HC:AAA", "HC:BBB"}
    assert hc[hc["player_id"] == "HC:AAA"].iloc[0]["proj"] == 25.5


def test_vor_column_computed(patched_sources):
    board = sb.build_season_board(2026, _league())
    assert "vor" in board.columns
    # Board is sorted by VOR descending (compute_vor guarantee).
    assert board["vor"].is_monotonic_decreasing


def test_adp_defaults_fill_missing(patched_sources):
    board = sb.build_season_board(2026, _league())
    # LB1 has no FFC ADP -> Sleeper adp_idp (40) floored to the IDP floor (150).
    lb = board[board["player_id"] == "LB1"].iloc[0]
    assert lb["adp"] == sb._IDP_ADP_FLOOR
    # HC rows get the static HC ADP.
    hc = board[board["player_id"] == "HC:AAA"].iloc[0]
    assert hc["adp"] == sb._HC_ADP


def test_fallback_when_sources_empty(monkeypatch):
    """Both projection sources empty -> prior-season pseudo-projection path."""
    empty = pd.DataFrame(columns=["player_id", "name", "position", "team"])
    monkeypatch.setattr(sb, "_safe_espn", lambda season, league, refresh: empty.copy())
    monkeypatch.setattr(sb, "_safe_sleeper", lambda season, refresh: empty.copy())
    monkeypatch.setattr(sb, "return_points_overlay", lambda league, season, **kw: {})
    monkeypatch.setattr(sb, "hc_draft_ev", lambda league, season: pd.DataFrame([
        {"player_id": "HC:AAA", "team": "AAA", "coach_label": "HC AAA",
         "position": "HC", "win_prob": 0.6, "expected_season_points": 17.0},
    ]))
    monkeypatch.setattr(sb, "load_ffc_adp",
                        lambda season, teams=12, fmt="ppr": pd.DataFrame(
                            columns=["player_id", "adp", "sd"]))

    prior = pd.DataFrame([
        {"player_id": "P1", "player_display_name": "Prior Guy", "position": "RB",
         "season": 2025, "games": 17, "pts": 250.0, "ppg": 14.7,
         "pts_nflverse_ppr": 250.0},
        {"player_id": "P2", "player_display_name": "Prior Two", "position": "WR",
         "season": 2025, "games": 16, "pts": 180.0, "ppg": 11.2,
         "pts_nflverse_ppr": 180.0},
    ])
    monkeypatch.setattr("fantasy.data.nfl.season_totals",
                        lambda seasons, engine, **kw: prior.copy())

    board = sb.build_season_board(2026, _league())
    assert board["proj_source"].iloc[0] == "prior_season"
    assert "P1" in set(board["player_id"])
    assert (board["position"] == "HC").sum() == 1
    assert "vor" in board.columns


# ── regressions: adversarial review findings ─────────────────────────────────
def test_espn_defender_rows_do_not_shadow_sleeper_idp(patched_sources, monkeypatch):
    """ESPN's kona feed lists defenders but projects NO IDP stats. Those empty
    rows must not enter the offense merge, where dedup would keep them and
    shadow Sleeper's real IDP projection with a 0."""
    espn_with_def = pd.DataFrame([
        {"player_id": "WR1", "name": "Alpha WR", "position": "WR", "team": "AAA",
         "receiving_targets": 100.0, "receptions": 80.0, "receiving_yards": 1200.0,
         "receiving_tds": 8.0},
        # Stat-less defender row straight from kona.
        {"player_id": "LB1", "name": "Mike Backer", "position": "LB", "team": "CCC"},
    ])
    monkeypatch.setattr(sb, "_safe_espn", lambda season, league, refresh: espn_with_def.copy())

    board = sb.build_season_board(2026, _league())
    lb = board[board["player_id"] == "LB1"]
    assert len(lb) == 1
    # Scored from Sleeper's stats: 90*1.0 + 40*0.5 + 3*2.0 = 116, not 0.
    assert lb.iloc[0]["proj"] == pytest.approx(116.0, abs=0.1)


def test_unslotted_positions_are_dropped_from_board(patched_sources, monkeypatch):
    """A position with no starting slot (D/ST here) has a zero replacement rank;
    leaving it on the board would give it an inflated VOR of proj-minus-nothing."""
    espn_with_dst = pd.DataFrame([
        {"player_id": "WR1", "name": "Alpha WR", "position": "WR", "team": "AAA",
         "receiving_targets": 100.0, "receptions": 80.0, "receiving_yards": 1200.0,
         "receiving_tds": 8.0},
        {"player_id": "DST:AAA", "name": "AAA D/ST", "position": "D/ST", "team": "AAA"},
    ])
    monkeypatch.setattr(sb, "_safe_espn", lambda season, league, refresh: espn_with_dst.copy())

    board = sb.build_season_board(2026, _league())
    assert "DST:AAA" not in set(board["player_id"])  # league starts no D/ST
    assert "WR1" in set(board["player_id"])


def test_overlay_owned_return_stats_never_double_count(patched_sources, monkeypatch):
    """If a projection source starts publishing return yardage, row scoring must
    ignore it — the returner overlay is the single owner of those points."""
    espn_with_returns = pd.DataFrame([
        {"player_id": "WR1", "name": "Alpha WR", "position": "WR", "team": "AAA",
         "receiving_targets": 100.0, "receptions": 80.0, "receiving_yards": 1200.0,
         "receiving_tds": 8.0,
         "kickoff_return_yards": 800.0},  # would be +200 pts if double-counted
    ])
    monkeypatch.setattr(sb, "_safe_espn", lambda season, league, refresh: espn_with_returns.copy())

    board = sb.build_season_board(2026, _league())
    wr = board[board["player_id"] == "WR1"].iloc[0]
    # targets 25 + rec 40 + yards mean(1200,1000)*0.1=110 + TDs 48 = 223,
    # plus the overlay's 200 return_pts — and nothing more.
    assert wr["return_pts"] == pytest.approx(200.0)
    assert wr["proj"] == pytest.approx(423.0, abs=0.5)
