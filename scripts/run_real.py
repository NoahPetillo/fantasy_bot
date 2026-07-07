"""Run one advise-only cycle on the REAL connected league, using ESPN's
projections as the primary input. No writes — recommendations only.

    uv run python scripts/run_real.py [week]
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

from fantasy.espn.client import EspnClient
from fantasy.league_state import build_live_snapshot
from fantasy.notify.console import ConsoleNotifier
from fantasy.orchestrator.cycle import run_cycle
from fantasy.orchestrator.store import Store
from fantasy.projections.service import ProjectionService

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
SEASON = 2025
from fantasy.projections.service import default_train_seasons
TRAIN = default_train_seasons(SEASON)


def main() -> int:
    week = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    client = EspnClient(season=SEASON)
    league = client.league_settings()
    print(f"League: {league.summary()}")
    print(f"Training model on {TRAIN}; using ESPN projections for week {week}...\n")

    service = ProjectionService(league).fit(TRAIN)
    espn_proj = client.week_projections(week)
    print(f"Fetched {len(espn_proj)} ESPN player projections for week {week}.")

    board = service.project(SEASON, week, espn_proj=espn_proj)
    src = board["proj_source"].value_counts().to_dict()
    print(f"Board: {len(board)} players, projection source = {src}\n")

    snap = build_live_snapshot(client, league, SEASON, week)
    me = snap.team_names.get(snap.my_team_id, snap.my_team_id)
    print(f"Your team: {me} ({len(snap.my_roster())} players)\n")
    print("════════ Recommendations (advise-only, ESPN-primary projections) ════════")

    store = Store(Path(tempfile.mkdtemp()) / "real.sqlite")
    fresh = run_cycle(service, league, SEASON, week, snapshot=snap, store=store,
                      notifier=ConsoleNotifier(), espn_proj=espn_proj)
    print(f"\n→ {len(fresh)} recommendations generated for {me} (week {week}).")
    print("Note: rosters reflect the league's CURRENT (end-of-2025) state; in-season this matches the week.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
