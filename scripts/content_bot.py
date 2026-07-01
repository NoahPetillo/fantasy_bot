"""Personal league content bot — standalone, single-tenant, self-contained.

Runs the content engine for ONE league (yours) off its own config
(``fantasy.moments.config``) + its own SQLite store, independent of the
multi-tenant web app. The SaaS refactor can reshape ``fantasy/config.py``, the
``Store``, and per-user ESPN handling without touching this.

Typical use — Tuesday morning, one command:

    uv run python scripts/content_bot.py            # scan, show the captions, ask, then post
    uv run python scripts/content_bot.py --yes       # scan + post all, no prompt (cron)
    uv run python scripts/content_bot.py --dry-run    # scan + print, never post
    uv run python scripts/content_bot.py --week 14    # recap a specific week
    uv run python scripts/content_bot.py --schedule   # long-running: Tue 10am + every 6h ET

A manual run PREVIEWS the moments (captions + image paths), then asks
"Post all N to Discord? [y/N]" — nothing hits the group chat until you say so.
``--yes`` skips the prompt (cron); ``--dry-run`` never posts; ``--schedule`` runs
the schedule and auto-posts.

Idempotency = POSTED-once: a moment is only remembered once it actually posts, so
previewing or cancelling never "uses it up". Persisted in its OWN file
(data/content_bot.sqlite), separate from the app's store.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import sys
import tempfile

from fantasy.moments.config import content_config as cfg

log = logging.getLogger("content_bot")


def _client():
    from fantasy.espn.client import EspnClient

    return EspnClient(league_id=cfg.espn_league_id, season=cfg.espn_season,
                      espn_s2=cfg.espn_s2, swid=cfg.espn_swid_braced)


def _posted_store():
    """The durable store — holds ONLY moments that were actually posted."""
    from fantasy.orchestrator.store import Store

    return Store(path=cfg.data_dir / "content_bot.sqlite")


def _scratch_store():
    """A throwaway store for generation, so preview/cancel never consume a moment."""
    from fantasy.orchestrator.store import Store

    fd, path = tempfile.mkstemp(prefix="content_bot_scratch_", suffix=".sqlite")
    os.close(fd)
    return Store(path=path), path


def _current_week(client) -> int:
    try:
        return int(getattr(client.league(), "current_week", 1) or 1)
    except Exception as e:  # noqa: BLE001
        log.warning("Could not read current week (%s); defaulting to 1.", e)
        return 1


def _already_posted(posted_store, p) -> bool:
    existing = posted_store.by_key(p.idempotency_key)
    return existing is not None and existing.status.value == "executed"


def _print_moment(i: int, n: int, p) -> None:
    caption = p.payload.get("caption") or p.title
    img = p.payload.get("image_path")
    print(f"\n[{i}/{n}] ({p.value:.0f} spice) {p.title}")
    print(f"    {caption}")
    if img:
        print(f"    🖼  {img}")


def _post_all(proposals: list, posted_store) -> int:
    from fantasy.moments.publisher import publish_moment
    from fantasy.orchestrator.models import ProposalStatus

    posted = 0
    for p in proposals:
        ref = publish_moment(p)
        if ref:
            posted_store.add(p)  # remember it ONLY now that it's live → never reposts
            posted_store.set_status(p.id, ProposalStatus.executed, ref)
            posted += 1
            print(f"    ✅ posted {p.title[:52]} ({ref})")
        else:
            print(f"    ⚠️  failed to post {p.title[:52]} — check DISCORD_WEBHOOK_URL")
    return posted


def run_once(week: int | None = None, do_content: bool = True, do_activity: bool = True,
             mode: str = "confirm") -> None:
    """mode: 'confirm' (preview + ask), 'auto' (post all), 'preview' (never post)."""
    if not cfg.has_espn_auth:
        log.error("No ESPN cookies (ESPN_S2/ESPN_SWID/ESPN_LEAGUE_ID) — nothing to do.")
        return

    cfg.content_autopost = False  # the bot owns the post decision, not the cycle
    from fantasy.moments.cycle import activity_cycle, content_cycle

    client = _client()
    posted_store = _posted_store()
    scratch, scratch_path = _scratch_store()  # generate here; discarded after
    season = client.season
    candidates: list = []
    try:
        if do_content:
            wk = week or _current_week(client)
            log.info("Weekly scan: season %s, recap week ~%s", season, wk)
            candidates += content_cycle(client, season, wk, store=scratch, notifier=None)
        if do_activity:
            log.info("Activity scan: trades + waivers")
            candidates += activity_cycle(client, season, store=scratch, notifier=None)
    finally:
        scratch.close()
        with contextlib.suppress(OSError):
            os.unlink(scratch_path)

    fresh = [p for p in candidates if not _already_posted(posted_store, p)]
    if not fresh:
        print("\nNothing new to post (already posted, or no spicy moments this week).")
        return

    n = len(fresh)
    print(f"\n=== {n} moment(s) ===")
    for i, p in enumerate(fresh, 1):
        _print_moment(i, n, p)

    if mode == "preview":
        print("\n(dry run — posted nothing. Re-run without --dry-run to send.)")
        return
    if mode == "confirm":
        if not sys.stdin.isatty():
            print("\nNon-interactive shell — not posting. Use --yes to post without a prompt.")
            return
        if input(f"\nPost all {n} to Discord? [y/N]: ").strip().lower() not in ("y", "yes"):
            print("Cancelled — nothing posted (these will show again next run).")
            return

    print(f"\nPosting {n} to Discord…")
    print(f"Done — {_post_all(fresh, posted_store)}/{n} posted.")


def run_scheduled() -> None:
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger

    sched = BlockingScheduler(timezone="America/New_York")
    sched.add_job(lambda: run_once(do_content=True, do_activity=False, mode="auto"),
                  CronTrigger(day_of_week="tue", hour=10), id="content_weekly")
    sched.add_job(lambda: run_once(do_content=False, do_activity=True, mode="auto"),
                  CronTrigger(hour="*/6"), id="content_activity")
    log.info("Content bot running: weekly recap Tue 10:00 ET, trades/waivers every 6h. Ctrl-C to stop.")
    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("Content bot stopped.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Personal league content bot (standalone).")
    ap.add_argument("--schedule", action="store_true",
                    help="run continuously on the ET schedule (auto-posts)")
    ap.add_argument("--yes", action="store_true", help="post all without the confirm prompt")
    ap.add_argument("--dry-run", action="store_true", help="generate + print, never post")
    ap.add_argument("--week", type=int, default=None, help="recap a specific week")
    ap.add_argument("--no-activity", action="store_true", help="skip the trades/waivers scan")
    args = ap.parse_args()

    if args.schedule:
        run_scheduled()
        return
    mode = "preview" if args.dry_run else "auto" if args.yes else "confirm"
    run_once(week=args.week, do_activity=not args.no_activity, mode=mode)


if __name__ == "__main__":
    main()
