"""Phase 4 demo — replay YOUR real 2025 draft + a self-play sanity check.

For each of your actual 2025 picks, show what the engine would have recommended
and whether that player outscored who you actually took (by realized 2025 points).
Then a quick self-play A/B vs ADP-autopick.

    uv run python scripts/draft_replay.py
"""

from __future__ import annotations

import logging

import pandas as pd

from fantasy.config import settings
from fantasy.draft.validate import replay_real_draft, run_selfplay
from fantasy.espn.client import EspnClient

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
pd.set_option("display.width", 160)
SEASON = 2025


def main() -> int:
    client = EspnClient(season=SEASON)
    league = client.league_settings()
    print(f"League: {league.summary()}\n")

    print(f"════════ Replaying your real {SEASON} draft (team {settings.espn_team_id}) ════════")
    table = replay_real_draft(client, SEASON, league)
    if table.empty:
        print("No picks found for your team.")
        return 0
    print(table[["round", "overall", "actual", "actual_pts", "engine", "engine_pts", "delta"]]
          .to_string(index=False))
    eng_roster = table.attrs.get("engine_roster_pts", 0.0)
    act_roster = table.attrs.get("actual_roster_pts", 0.0)
    wins = (table["delta"] > 0).sum()
    print(f"\nBest season-long lineup from each drafted roster (realized 2025 pts):")
    print(f"  your actual draft: {act_roster:.0f}   |   engine's draft: {eng_roster:.0f}   "
          f"({eng_roster - act_roster:+.0f})")
    print(f"  engine's pick beat your actual pick at {wins}/{len(table)} slots.")
    print("(Realized points are hindsight — both rosters judged by the same yardstick.)")

    print("\n════════ Self-play A/B vs ADP-autopick (2022-2024) ════════")
    res = run_selfplay(league, seasons=[2022, 2023, 2024], n_per_season=12)
    lo, hi = res["ci95"]
    print(f"Our agent vs ADP-autopick at the same seat: "
          f"mean +{res['mean_delta']:.1f} realized pts/draft "
          f"(95% CI [{lo:+.1f}, {hi:+.1f}]), win rate {res['win_rate']*100:.0f}% "
          f"over {res['n']} drafts.")
    print("Honest read: drafting edges are small vs strong ADP; value shows in avoiding reaches.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
