"""Build the dashboard snapshot from your real league, then serve it.

    uv run python scripts/dashboard.py [week]          # build + serve at :8000
    uv run python scripts/dashboard.py --build-only 12 # just refresh the snapshot

Open http://127.0.0.1:8000 once it's running.
"""

from __future__ import annotations

import logging
import sys

from fantasy.api.dashboard_data import assemble, write_snapshot
from fantasy.config import settings
from fantasy.espn.client import EspnClient
from fantasy.orchestrator.store import Store
from fantasy.projections.service import ProjectionService

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
SEASON = 2025
TRAIN = [2021, 2022, 2023, 2024]


def build(week: int) -> None:
    client = EspnClient(season=SEASON)
    league = client.league_settings()
    print(f"League: {league.summary()}\nTraining model + assembling week {week} snapshot...")
    service = ProjectionService(league).fit(TRAIN)
    store = Store(settings.db_path)
    payload = assemble(service, league, store, SEASON, week, client=client)
    write_snapshot(payload)
    print(f"✓ Snapshot written: {len(payload['waivers'])} waivers, {len(payload['trades'])} trades, "
          f"{len(payload['lineup'])} starters, {len(payload['standings'])} teams, "
          f"{len(payload['feed'])} feed items.")


def main() -> int:
    args = [a for a in sys.argv[1:] if a != "--build-only"]
    week = int(args[0]) if args else 12
    build(week)
    if "--build-only" in sys.argv:
        return 0
    import uvicorn

    from fantasy.api.app import app
    print("\n→ Dashboard live at http://127.0.0.1:8000  (Ctrl-C to stop)")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
