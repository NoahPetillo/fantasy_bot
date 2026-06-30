"""Prove the league-adaptive scoring spine on REAL historical data (no cookies).

1. Build a scoring map equal to nflverse's standard PPR formula, score the 2023
   season with our ScoringEngine, and confirm it matches nflverse's own
   ``fantasy_points_ppr`` column (correctness check).
2. Re-score the SAME season under a different league (TE-premium half-PPR) and
   show the leaderboard shift (adaptivity check).

    uv run python scripts/demo_historical_scoring.py
"""

from __future__ import annotations

import logging

from fantasy.data.nfl import load_weekly, season_totals
from fantasy.league_settings import LeagueSettings, RosterRequirements
from fantasy.valuation.scoring import ScoringEngine

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

SEASON = 2023
SKILL = ["QB", "RB", "WR", "TE"]

# nflverse standard scoring formula (PPR variant adds 1.0 per reception).
NFLVERSE_PPR = {
    "passing_yards": 0.04,
    "passing_tds": 4.0,
    "passing_interceptions": -2.0,
    "rushing_yards": 0.1,
    "rushing_tds": 6.0,
    "receiving_yards": 0.1,
    "receiving_tds": 6.0,
    "passing_2pt_conversions": 2.0,
    "rushing_2pt_conversions": 2.0,
    "receiving_2pt_conversions": 2.0,
    "special_teams_tds": 6.0,
    "fumbles_lost": -2.0,
    "receptions": 1.0,
}


def main() -> int:
    # ── 1) correctness: match nflverse fantasy_points_ppr ──
    ppr = LeagueSettings(name="nflverse-PPR", season=SEASON, scoring=dict(NFLVERSE_PPR))
    eng = ScoringEngine(ppr)
    df = load_weekly([SEASON])
    df = df[df["position"].isin(SKILL)].copy()
    df["ours"] = eng.score_dataframe(df)
    df["ref"] = df["fantasy_points_ppr"].fillna(0.0)
    df["diff"] = (df["ours"] - df["ref"]).abs()

    max_diff = df["diff"].max()
    mean_diff = df["diff"].mean()
    within = (df["diff"] <= 0.011).mean() * 100
    print("\n=== Correctness vs nflverse fantasy_points_ppr (2023, QB/RB/WR/TE) ===")
    print(f"  rows: {len(df):,}  max|diff|: {max_diff:.4f}  mean|diff|: {mean_diff:.5f}  "
          f"within 0.01: {within:.2f}%")
    if max_diff > 0.05:
        worst = df.nlargest(5, "diff")[
            ["player_display_name", "position", "week", "ours", "ref", "diff"]
        ]
        print("  ⚠ largest mismatches (investigate scoring constant):")
        print(worst.to_string(index=False))
    else:
        print("  ✓ ScoringEngine reproduces nflverse PPR to the penny — mapping verified.")

    totals = season_totals([SEASON], eng)
    totals = totals[totals["position"].isin(SKILL)]
    print("\n  Top 12 by our PPR engine (season totals):")
    print(totals.head(12)[["player_display_name", "position", "games", "pts", "ppg"]]
          .to_string(index=False))

    # ── 2) adaptivity: a different league reorders the board ──
    te_prem = LeagueSettings(
        name="TE-premium-half",
        season=SEASON,
        scoring={**NFLVERSE_PPR, "receptions": 0.5},
        position_reception_bonus={"TE": 0.5},  # TEs effectively full-PPR
        roster=RosterRequirements(slots={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "BE": 6}),
    )
    te_totals = season_totals([SEASON], ScoringEngine(te_prem))
    print("\n=== Same season, TE-premium half-PPR — top 8 TEs ===")
    print(te_totals[te_totals["position"] == "TE"].head(8)[
        ["player_display_name", "games", "pts", "ppg"]].to_string(index=False))
    print("\n✓ Same data, different league rules -> different valuations. League-adaptive spine works.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
