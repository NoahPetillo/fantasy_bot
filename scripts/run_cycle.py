"""Phase 2 demo — run one notify-and-approve cycle in DRY-RUN (no cookies).

Synthesizes a 12-team half-PPR league from the 2023 wk-10 value board, generates
start/sit + waiver + trade recommendations, logs them to SQLite, and pushes them
to the console notifier. Runs twice to show idempotency (no duplicate pings).

    uv run python scripts/run_cycle.py
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fantasy.league_settings import LeagueSettings, RosterRequirements
from fantasy.notify.console import ConsoleNotifier
from fantasy.orchestrator.cycle import run_cycle
from fantasy.orchestrator.store import Store
from fantasy.projections.service import ProjectionService

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

HALF_PPR = {
    "passing_yards": 0.04, "passing_tds": 4.0, "passing_interceptions": -2.0,
    "rushing_yards": 0.1, "rushing_tds": 6.0, "receiving_yards": 0.1, "receiving_tds": 6.0,
    "passing_2pt_conversions": 2.0, "rushing_2pt_conversions": 2.0, "receiving_2pt_conversions": 2.0,
    "special_teams_tds": 6.0, "fumbles_lost": -2.0, "receptions": 0.5,
}
LEAGUE = LeagueSettings(
    name="Demo 12-team half-PPR", team_count=12, scoring=dict(HALF_PPR),
    regular_season_weeks=14,
    roster=RosterRequirements(slots={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1,
                                     "K": 1, "D/ST": 1, "BE": 6}),
)
SEASON, WEEK = 2023, 10


def main() -> int:
    print(f"League: {LEAGUE.summary()}\nFitting projection model (train 2020-2022)...\n")
    service = ProjectionService(LEAGUE).fit([2020, 2021, 2022])

    db = Path(tempfile.mkdtemp()) / "cycle_demo.sqlite"
    store = Store(db)
    notifier = ConsoleNotifier()

    print(f"════════ CYCLE 1 — {SEASON} week {WEEK} (dry-run) ════════")
    fresh = run_cycle(service, LEAGUE, SEASON, WEEK, store=store, notifier=notifier)
    print(f"\n→ {len(fresh)} new proposals logged.\n")

    print("════════ CYCLE 2 — same inputs (idempotency check) ════════")
    fresh2 = run_cycle(service, LEAGUE, SEASON, WEEK, store=store, notifier=notifier)
    print(f"→ {len(fresh2)} new proposals (expected 0 — duplicates suppressed).")

    counts = {}
    for p in store.list(limit=500):
        counts[p.kind.value] = counts.get(p.kind.value, 0) + 1
    print(f"\nAction log totals by kind: {counts}")
    print("✓ Phase 2 loop works: project → recommend → log → notify → (await approval).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
