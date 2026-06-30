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

    def content_now(self):
        """Scan the just-finished week for hype-worthy moments → approve-to-post.

        Needs box scores, so it's a no-op without a live ESPN client (cookies).
        """
        if self.client is None:
            log.info("Content scan skipped: no ESPN client (box scores need cookies).")
            return []
        from fantasy.moments.cycle import content_cycle

        season, week = self._season_week()
        return content_cycle(self.client, season, week, store=self.store, notifier=self.notifier)

    def activity_now(self):
        """Scan the transaction feed for completed trades + notable FAAB pickups.

        No-op without a live client; the feed itself is current-season only and
        degrades gracefully off-season.
        """
        if self.client is None:
            log.info("Activity scan skipped: no ESPN client (needs cookies).")
            return []
        from fantasy.moments.cycle import activity_cycle

        season, _ = self._season_week()
        return activity_cycle(self.client, season, store=self.store, notifier=self.notifier)

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
        # League content recap: Tue 10am ET, after MNF + stat corrections settle.
        sched.add_job(self.content_now, CronTrigger(day_of_week="tue", hour=10),
                      id="content_scan", replace_existing=True)
        # Transaction feed (trades/waivers): every 6h — trades lag the review window
        # anyway, so there's no point polling faster.
        sched.add_job(self.activity_now, CronTrigger(hour="*/6"),
                      id="content_activity_scan", replace_existing=True)
        sched.start()
        log.info("Scheduler started: news(30m), trade(daily 9am), waiver(Tue 8pm), "
                 "lineup(Thu/Sun/Mon 11am), content(Tue 10am), activity(every 6h) ET")
        return sched
