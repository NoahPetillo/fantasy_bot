"""Personal league content bot — standalone, single-tenant, self-contained.

Runs the content engine for ONE league (yours) off its own config
(``fantasy.moments.config``, env-driven) and its own SQLite idempotency store,
completely independent of the multi-tenant web app. This is the durable "just for
me" lane: the SaaS refactor can reshape ``fantasy/config.py``, the ``Store``, and
per-user ESPN handling without touching this bot.

Usage:
    uv run python scripts/content_bot.py --once             # one scan now (cron-friendly)
    uv run python scripts/content_bot.py --once --week 14    # recap a specific week
    uv run python scripts/content_bot.py --once --dry-run    # generate, print, DON'T post
    uv run python scripts/content_bot.py --schedule          # long-running: Tue 10am + every 6h ET

Posting respects CONTENT_AUTOPOST. For the hands-off experience set
CONTENT_AUTOPOST=true so spicy-enough moments post straight to Discord; otherwise
they're generated + stored and you'd approve them elsewhere.

Config comes from the same env vars as before (ESPN_S2/SWID/LEAGUE_ID/SEASON,
DISCORD_WEBHOOK_URL, GROQ_API_KEY, CONTENT_*). Idempotency lives in its OWN file
(data/content_bot.sqlite), separate from the app's store.
"""

from __future__ import annotations

import argparse
import logging

from fantasy.moments.config import content_config as cfg

log = logging.getLogger("content_bot")


def _client():
    from fantasy.espn.client import EspnClient

    return EspnClient(league_id=cfg.espn_league_id, season=cfg.espn_season,
                      espn_s2=cfg.espn_s2, swid=cfg.espn_swid_braced)


def _store():
    from fantasy.orchestrator.store import Store

    # Dedicated DB file — never collides with the app's store (which becomes Postgres).
    return Store(path=cfg.data_dir / "content_bot.sqlite")


def _current_week(client) -> int:
    try:
        return int(getattr(client.league(), "current_week", 1) or 1)
    except Exception as e:  # noqa: BLE001
        log.warning("Could not read current week (%s); defaulting to 1.", e)
        return 1


def _report(label: str, fresh: list, dry_run: bool) -> None:
    verb = "would post" if dry_run else "posted/queued"
    log.info("[%s] %d new moment(s) — %s", label, len(fresh), verb)
    for p in fresh:
        caption = p.payload.get("caption") or p.title
        print(f"  • ({p.value:.0f}) {p.title}\n      {caption}\n")


def run_once(week: int | None = None, do_content: bool = True,
             do_activity: bool = True, dry_run: bool = False) -> None:
    if not cfg.has_espn_auth:
        log.error("No ESPN cookies (ESPN_S2/ESPN_SWID/ESPN_LEAGUE_ID) — nothing to do.")
        return
    if dry_run:
        cfg.content_autopost = False  # generate + print, never post
    elif cfg.content_autopost and not cfg.discord_webhook_url:
        log.warning("CONTENT_AUTOPOST is on but DISCORD_WEBHOOK_URL is unset — nothing will post.")

    from fantasy.moments.cycle import activity_cycle, content_cycle

    client, store = _client(), _store()
    season = client.season
    if do_content:
        wk = week or _current_week(client)
        log.info("Weekly scan: season %s, recap week ~%s (autopost=%s, dry_run=%s)",
                 season, wk, cfg.content_autopost, dry_run)
        _report("weekly", content_cycle(client, season, wk, store=store, notifier=None), dry_run)
    if do_activity:
        log.info("Activity scan: trades + waivers")
        _report("activity", activity_cycle(client, season, store=store, notifier=None), dry_run)


def run_scheduled() -> None:
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger

    sched = BlockingScheduler(timezone="America/New_York")
    sched.add_job(lambda: run_once(do_content=True, do_activity=False),
                  CronTrigger(day_of_week="tue", hour=10), id="content_weekly")
    sched.add_job(lambda: run_once(do_content=False, do_activity=True),
                  CronTrigger(hour="*/6"), id="content_activity")
    log.info("Content bot running: weekly recap Tue 10:00 ET, trades/waivers every 6h. Ctrl-C to stop.")
    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("Content bot stopped.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Personal league content bot (standalone).")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true", help="run a single scan now")
    mode.add_argument("--schedule", action="store_true", help="run continuously on the ET schedule")
    ap.add_argument("--week", type=int, default=None, help="recap a specific week (with --once)")
    ap.add_argument("--no-activity", action="store_true", help="skip the trades/waivers scan")
    ap.add_argument("--dry-run", action="store_true", help="generate + print, but do not post")
    args = ap.parse_args()

    if args.schedule:
        run_scheduled()
    else:
        run_once(week=args.week, do_activity=not args.no_activity, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
