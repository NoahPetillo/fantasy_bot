"""VOR engine: replacement level must track league roster settings."""

from __future__ import annotations

import pandas as pd

from fantasy.league_settings import LeagueSettings, RosterRequirements
from fantasy.valuation.vor import compute_vor, replacement_counts


def _league(team_count=12, **slots):
    return LeagueSettings(team_count=team_count, roster=RosterRequirements(slots=slots))


def test_replacement_counts_scale_with_teams_and_slots():
    std = _league(12, QB=1, RB=2, WR=2, TE=1, FLEX=1)
    counts = replacement_counts(std)
    assert counts["QB"] == 12  # 1 per team
    assert counts["TE"] == 12 + round(12 * (1 / 3))  # 1 dedicated + 1/3 flex share
    # Bigger league -> deeper replacement level.
    big = _league(14, QB=1, RB=2, WR=2, TE=1, FLEX=1)
    assert replacement_counts(big)["RB"] > counts["RB"]


def test_superflex_deepens_qb_replacement():
    one = replacement_counts(_league(12, QB=1, RB=2, WR=2, TE=1, FLEX=1))
    sf = replacement_counts(_league(12, QB=1, RB=2, WR=2, TE=1, FLEX=1, OP=1))
    assert sf["QB"] > one["QB"]  # superflex makes QBs much scarcer


def test_two_flex_idp_hc_replacement_counts():
    """12-team, 2-FLEX, +DP +HC roster -> RB/WR≈32, TE=20, DP=12, HC=12."""
    ls = _league(12, QB=1, RB=2, WR=2, TE=1, FLEX=2, K=1, DP=1, HC=1)
    counts = replacement_counts(ls)
    # RB/WR: 2 dedicated + 2 flex slots * 1/3 share each = 2 + 2/3 -> *12 = 32.
    assert counts["RB"] == 32
    assert counts["WR"] == 32
    # TE: 1 dedicated + 2/3 flex share -> (1 + 2/3)*12 = 20.
    assert counts["TE"] == 20
    # Cross-position pools sized by team count.
    assert counts["DP"] == 12
    assert counts["HC"] == 12


def test_dp_pool_baseline_across_mixed_positions():
    """DP replacement baseline is computed across a mixed LB/S/DE pool."""
    ls = _league(12, QB=1, RB=2, WR=2, TE=1, FLEX=2, DP=1, HC=1)
    # 30 mixed defenders across LB/S/DE, descending projections.
    positions = (["LB"] * 12 + ["S"] * 10 + ["DE"] * 8)
    proj = pd.DataFrame({
        "position": positions,
        "proj": list(range(200, 200 - len(positions), -1)),
    })
    proj["player_id"] = [f"d{i}" for i in range(len(proj))]
    out = compute_vor(proj, ls)
    # Every defender shares ONE replacement level (the DP pool), regardless of pos.
    assert out["replacement"].nunique() == 1
    baseline = out["replacement"].iloc[0]
    assert baseline > 0
    # The best defender is above replacement; the worst is below.
    assert out["vor"].max() > 0 > out["vor"].min()


def test_compute_vor_orders_and_signs():
    proj = pd.DataFrame(
        {
            "position": ["QB"] * 14 + ["RB"] * 40,
            "proj": list(range(300, 300 - 14, -1)) + list(range(250, 250 - 40, -1)),
        }
    )
    proj["player_id"] = range(len(proj))
    out = compute_vor(proj, _league(12, QB=1, RB=2, WR=2, TE=1, FLEX=1))
    # Top row has the highest VOR; some low-ranked players are below replacement.
    assert out.iloc[0]["vor"] == out["vor"].max()
    assert (out["vor"] < 0).any()
    # VOR is sorted descending.
    assert out["vor"].is_monotonic_decreasing
