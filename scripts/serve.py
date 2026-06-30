"""Run the Phase 2 service: FastAPI approval API + (optional) cadence scheduler.

    uv run python scripts/serve.py

With ESPN cookies in .env it connects to your real league, trains the model on
recent seasons, reads your exact scoring/roster from mSettings, and schedules the
advise-only cycles. Without cookies it serves the API + control panel over
whatever is already in the action log (use scripts/run_cycle.py to populate it).
"""

from __future__ import annotations

import logging

import uvicorn

from fantasy.api.app import app
from fantasy.config import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def bootstrap_orchestrator():
    """Build the live orchestrator if cookies are present; else None."""
    if not settings.has_espn_auth:
        logging.warning("No ESPN cookies — serving API/control-panel only (no live scheduler). "
                        "Add cookies to .env to enable live cycles.")
        return None
    from fantasy.espn.client import EspnClient
    from fantasy.notify.base import get_notifier
    from fantasy.orchestrator.scheduler import Orchestrator
    from fantasy.projections.service import ProjectionService

    client = EspnClient()
    league = client.league_settings()
    logging.info("Live league: %s", league.summary())
    season = settings.espn_season
    train = [season - 4, season - 3, season - 2, season - 1]
    service = ProjectionService(league).fit(train)
    orch = Orchestrator(league, service, client=client, notifier=get_notifier())
    return orch


@app.on_event("startup")
async def _startup():
    orch = bootstrap_orchestrator()
    if orch is not None:
        orch.start()


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
