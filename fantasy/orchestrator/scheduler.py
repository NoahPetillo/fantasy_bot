"""Cadence scheduler — runs decision cycles on a schedule (APScheduler).

Cadence (advise/notify-only in Phase 2):
- trade scan        : daily
- waiver prep       : the evening before waivers process
- lineup guard      : game-day mornings (pre-lock; exact per-kickoff timing later)

Each job builds a fresh snapshot (live from ESPN when cookies are present, else a
dry-run snapshot) and calls run_cycle, which notifies only on NEW proposals.
NOTE: lineup-lock timing must respect ET↔UTC/DST — a missed lock is permanent.
"""

from __future__ import annotations

import logging

from fantasy.config import settings
from fantasy.league_settings import LeagueSettings
from fantasy.league_state import build_dryrun_snapshot, build_live_snapshot
from fantasy.orchestrator.cycle import run_cycle
from fantasy.orchestrator.store import Store
from fantasy.projections.service import ProjectionService

log = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, league: LeagueSettings, service: ProjectionService,
                 client=None, store: Store | None = None, notifier=None):
        self.league = league
        self.service = service
        self.client = client  # EspnClient (live) or None (dry-run)
        self.store = store or Store()
        self.notifier = notifier

    def _season_week(self) -> tuple[int, int]:
        if self.client is not None:
            try:
                lg = self.client.league()
                return self.client.season, int(getattr(lg, "current_week", 1) or 1)
            except Exception as e:  # noqa: BLE001
                log.warning("Could not read current week from ESPN: %s", e)
        return settings.espn_season, 1

    def _snapshot(self, season, week, board):
        if self.client is not None:
            return build_live_snapshot(self.client, self.league, season, week)
        return build_dryrun_snapshot(board, self.league, season, week)

    def _espn_proj(self, week):
        if self.client is None:
            return None
        try:
            return self.client.week_projections(week)
        except Exception as e:  # noqa: BLE001
            log.warning("Could not fetch ESPN projections: %s", e)
            return None

    def cycle_now(self, kinds=("start_sit", "waiver", "trade")):
        season, week = self._season_week()
        espn_proj = self._espn_proj(week)
        board = self.service.project(season, week, espn_proj=espn_proj)
        if board.empty:
            log.warning("No board for %s wk%s; skipping cycle.", season, week)
            return []
        snap = self._snapshot(season, week, board)
        return run_cycle(self.service, self.league, season, week, snapshot=snap,
                         store=self.store, notifier=self.notifier, kinds=kinds,
                         espn_proj=espn_proj)

    def news_now(self):
        """Scan news/injuries and raise alerts for the user's roster + waiver targets."""
        from fantasy.news.ingest import news_cycle

        season, week = self._season_week()
        board = self.service.project(season, week)
        if board.empty:
            return []
        snap = self._snapshot(season, week, board)
        return news_cycle(snap, store=self.store, notifier=self.notifier)

    def start(self):
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.cron import CronTrigger

        sched = AsyncIOScheduler(timezone="America/New_York")
        sched.add_job(lambda: self.cycle_now(("trade",)), CronTrigger(hour=9),
                      id="trade_scan", replace_existing=True)
        sched.add_job(lambda: self.cycle_now(("waiver",)), CronTrigger(day_of_week="tue", hour=20),
                      id="waiver_prep", replace_existing=True)
        sched.add_job(lambda: self.cycle_now(("start_sit",)),
                      CronTrigger(day_of_week="thu,sun,mon", hour=11),
                      id="lineup_guard", replace_existing=True)
        # News/injury scan every 30 min (tighten to ~5 min pre-kickoff later).
        sched.add_job(self.news_now, CronTrigger(minute="*/30"),
                      id="news_scan", replace_existing=True)
        sched.start()
        log.info("Scheduler started: news(30m), trade(daily 9am), waiver(Tue 8pm), "
                 "lineup(Thu/Sun/Mon 11am) ET")
        return sched
