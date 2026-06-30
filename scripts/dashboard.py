"""Build dashboard snapshots from your registered leagues, then serve.

    uv run python scripts/dashboard.py [week]               # build all leagues + serve
    uv run python scripts/dashboard.py --build-only [week]  # just refresh snapshots
    uv run python scripts/dashboard.py --league 12345 13    # one league, week 13

Leagues come from data/leagues.json (auto-seeded from your .env on first run);
add more from the dashboard's "Add League" form. Open http://127.0.0.1:8000.
"""

from __future__ import annotations

import logging
import sys

from fantasy.api.build import build_full
from fantasy.leagues import registry

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


def main() -> int:
    argv = sys.argv[1:]
    build_only = "--build-only" in argv
    league_id = None
    if "--league" in argv:
        i = argv.index("--league")
        league_id = int(argv[i + 1])
        del argv[i:i + 2]
    positional = [a for a in argv if not a.startswith("--")]
    week = int(positional[0]) if positional else None

    registry().seed_default()
    refs = [registry().get(league_id)] if league_id is not None else registry().all()
    refs = [r for r in refs if r]
    if not refs:
        print("No leagues registered. Set ESPN_LEAGUE_ID/ESPN_TEAM_ID in .env, or add one in the UI.")
        return 1

    for ref in refs:
        print(f"Building league {ref.league_id} (team {ref.team_id}, {ref.season}) ...")
        try:
            payload = build_full(ref, week=week)
            print(f"  ✓ {payload['team']['name']}: {len(payload['waivers'])} waivers, "
                  f"{len(payload['trades'])} trades, report={'yes' if payload.get('report') else 'no'}.")
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ failed: {e}")

    if build_only:
        return 0
    import uvicorn

    from fantasy.api.app import app
    print("\n→ Dashboard live at http://127.0.0.1:8000  (Ctrl-C to stop)")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
