"""Phase 1 end-to-end on real data (no cookies):

  train on 2021-2024, project 2025, then:
  1. accuracy vs naive baselines (must beat trailing-4),
  2. a cross-positional VOR value board for the season,
  3. distribution floor/median/ceiling for the top players.

    uv run python scripts/backtest_projections.py
"""

from __future__ import annotations

import logging

import pandas as pd

from fantasy.backtest.projections_backtest import format_report, run_backtest
from fantasy.league_settings import LeagueSettings, RosterRequirements
from fantasy.projections.distributions import VarianceModel
from fantasy.valuation.scoring import ScoringEngine
from fantasy.valuation.vor import compute_vor, replacement_counts

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
pd.set_option("display.width", 140)

# A representative 12-team half-PPR league (stands in until real mSettings load).
HALF_PPR = {
    "passing_yards": 0.04, "passing_tds": 4.0, "passing_interceptions": -2.0,
    "rushing_yards": 0.1, "rushing_tds": 6.0, "receiving_yards": 0.1, "receiving_tds": 6.0,
    "passing_2pt_conversions": 2.0, "rushing_2pt_conversions": 2.0, "receiving_2pt_conversions": 2.0,
    "special_teams_tds": 6.0, "fumbles_lost": -2.0, "receptions": 0.5,
}
LEAGUE = LeagueSettings(
    name="Default 12-team half-PPR", team_count=12, scoring=dict(HALF_PPR),
    roster=RosterRequirements(slots={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1,
                                     "K": 1, "D/ST": 1, "BE": 7}),
)
TRAIN, TEST = [2021, 2022, 2023, 2024], 2025


def main() -> int:
    engine = ScoringEngine(LEAGUE)
    print(f"League: {LEAGUE.summary()}\n")

    report, model, blender = run_backtest(TRAIN, TEST, engine)
    print(format_report(report, TEST), "\n")
    print(blender.describe(), "\n")

    # ── season VOR board (sum blended weekly projections -> season value) ──
    from fantasy.data.nfl import load_weekly
    from fantasy.projections.features import build_features

    feat = build_features(load_weekly([*TRAIN, TEST]), engine)
    test = feat[feat["season"] == TEST].copy()
    test["proj"] = model.predict(test)
    test["proj"] = blender.predict_fast(test)
    season = (
        test.groupby(["player_id", "player_display_name", "position"], as_index=False)
        .agg(proj=("proj", "sum"), actual=("y", "sum"), games=("week", "nunique"))
    )
    board = compute_vor(season, LEAGUE)
    print("Replacement ranks (startable players league-wide):", replacement_counts(LEAGUE))
    print("\nTop 15 by projected season VOR (with actual points for reference):")
    print(board.head(15)[["player_display_name", "position", "games", "proj", "replacement",
                          "vor", "actual"]].round(1).to_string(index=False))

    # ── distribution: floor / median / ceiling for the board's top names ──
    vm = VarianceModel().fit(test)
    print("\nWeekly distribution (per-game floor/median/ceiling) for top 5:")
    top5 = board.head(5)
    for _, r in top5.iterrows():
        per_game = r["proj"] / max(r["games"], 1)
        q = vm.quantiles(r["position"], per_game, qs=(0.1, 0.5, 0.9))
        print(f"  {r['player_display_name']:22s} {r['position']:3s} "
              f"floor {q[0.1]:5.1f} | median {q[0.5]:5.1f} | ceiling {q[0.9]:5.1f}")
    print("\n✓ Phase 1 pipeline runs end-to-end: projections → VOR board → distributions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
