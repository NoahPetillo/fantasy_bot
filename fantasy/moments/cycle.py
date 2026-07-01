"""Content cycle: scan a finished week → rank moments → generate → propose.

Mirrors fantasy.news.ingest.news_cycle and the orchestrator's run_cycle: build
proposals, persist only NEW ones (idempotency by moment identity), and notify on
those. The difference is the proposal kind (``moment``) and that "executing" a
moment means posting it to the group chat — which happens later, on approval,
via fantasy.api.app.on_approved → fantasy.moments.publisher.

Content (caption + graphic) is generated HERE, before approval, so the human
reviews the exact post they're greenlighting. Generation is skipped for moments
already raised this week, so re-runs are cheap.
"""

from __future__ import annotations

import logging

from fantasy.moments.config import content_config as settings  # decoupled from the app
from fantasy.moments.activity import detect_trades, detect_waivers
from fantasy.moments.content import render_card, write_caption
from fantasy.moments.detector import detect_moments
from fantasy.moments.models import Moment
from fantasy.moments.score import rank_and_select
from fantasy.moments.standings import detect_rivalries, detect_streaks
from fantasy.orchestrator.models import Proposal, ProposalKind, ProposalStatus
from fantasy.orchestrator.store import Store

log = logging.getLogger(__name__)


def _scored(box) -> bool:
    return (getattr(box, "home_score", 0) or 0) > 0 or (getattr(box, "away_score", 0) or 0) > 0


def _resolve_box_scores(client, week: int) -> tuple[list, int]:
    """Find the most recent week that actually has scores (handles the Tue/Wed
    scoring-period rollover, where ``current_week`` may already point at an
    unplayed week)."""
    for w in (week, week - 1, week - 2):
        if w < 1:
            continue
        try:
            bs = list(client.box_scores(w) or [])
        except Exception as e:  # noqa: BLE001
            log.warning("box_scores(%s) failed: %s", w, e)
            continue
        if bs and any(_scored(b) for b in bs):
            return bs, w
    return [], week


def _moment_to_proposal(m: Moment) -> Proposal:
    """Shell proposal (identity + ranking) — caption/image are filled in after the
    idempotency check so duplicates don't trigger generation."""
    return Proposal(
        kind=ProposalKind.moment, season=m.season, week=m.week, team_id=m.team_id,
        title=m.headline, detail="", value=round(m.spice, 1), confidence=0.95,
        payload={"key_fields": m.key_fields(), "moment_type": m.type.value, "channel": "discord"},
    )


def _maybe_autopost(p: Proposal, store: Store) -> bool:
    """Hands-off mode: if auto-post is on and the moment clears the spice bar,
    post to Discord now (skipping human approval) and mark it executed. On failure
    (e.g. no webhook) returns False so it falls back to the approval path."""
    if not settings.content_autopost or p.value < settings.content_autopost_min_spice:
        return False
    from fantasy.moments.publisher import publish_moment

    ref = publish_moment(p)
    if ref:
        store.set_status(p.id, ProposalStatus.executed, ref)
        log.info("Auto-posted moment %s -> %s", p.id, ref)
        return True
    log.warning("Auto-post failed for %s; leaving it for manual approval.", p.id)
    return False


def _emit(moments: list[Moment], store: Store, notifier, per_week: int | None,
          generate: bool) -> list[Proposal]:
    """Rank → dedup → (generate caption+graphic) → persist → auto-post or notify."""
    fresh: list[Proposal] = []
    for m in rank_and_select(moments, n=per_week):
        p = _moment_to_proposal(m)
        if store.by_key(p.idempotency_key) is not None:
            continue  # already raised
        if generate:
            caption = write_caption(m)
            img = render_card(m)
            p.payload["caption"] = caption
            p.payload["image_path"] = str(img) if img else None
            p.detail = caption + (f"\n\n🖼  Image ready: {img}" if img else "\n\n(caption only)")
        if store.add(p):
            if not _maybe_autopost(p, store) and notifier is not None:
                ref = notifier.notify(p)
                if ref:
                    store.set_status(p.id, p.status, ref)
            fresh.append(p)
    return fresh


def _teams(client) -> list:
    try:
        return list(client.teams() or [])
    except Exception as e:  # noqa: BLE001
        log.warning("teams() failed (streaks/rivalries skipped): %s", e)
        return []


def _manager_map(teams) -> dict[int, str]:
    """team_id -> manager first name (from ESPN owners), for roasting by name."""
    out: dict[int, str] = {}
    for t in teams or []:
        tid = getattr(t, "team_id", None)
        for o in getattr(t, "owners", []) or []:
            fn = (o.get("firstName") if isinstance(o, dict) else "") or ""
            if tid is not None and fn.strip():
                out[tid] = fn.strip()
                break
    return out


def _enrich_managers(moments: list[Moment], teams: list) -> list[Moment]:
    mgrs = _manager_map(teams)
    for m in moments:
        if m.manager is None and m.team_id is not None:
            m.manager = mgrs.get(m.team_id)
    return moments


def content_cycle(client, season: int, week: int, store: Store | None = None,
                  notifier=None, per_week: int | None = None, generate: bool = True,
                  recap_week: int | None = None) -> list[Proposal]:
    """Weekly recap: box-score moments + standings moments (streaks, rivalries)."""
    box_scores, resolved_week = _resolve_box_scores(client, recap_week or week)
    if not box_scores:
        log.info("Content cycle: no scored week found near wk%s — nothing to do.", week)
        return []

    moments = detect_moments(box_scores, season, resolved_week)
    teams = _teams(client)
    if teams:
        moments += detect_streaks(teams, season, resolved_week, min_len=settings.content_streak_min)
        moments += detect_rivalries(teams, season, resolved_week, settings.content_rivalries)
    _enrich_managers(moments, teams)

    store = store or Store()
    fresh = _emit(moments, store, notifier, per_week, generate)
    log.info("Content cycle wk%s: %d moments, %d new (voice=%s).",
             resolved_week, len(moments), len(fresh), settings.content_voice)
    return fresh


def activity_cycle(client, season: int, store: Store | None = None, notifier=None,
                   per_scan: int | None = None, generate: bool = True,
                   size: int = 60) -> list[Proposal]:
    """Transaction recap: completed trades + notable FAAB pickups from the activity
    feed. The feed is a current-season endpoint and 404s off-season — handled here
    so the cycle is a safe no-op rather than an error."""
    try:
        activities = list(client.recent_activity(size=size) or [])
    except Exception as e:  # noqa: BLE001
        log.info("recent_activity unavailable (likely off-season / current-season only): %s", e)
        return []

    moments = detect_trades(activities, season)
    moments += detect_waivers(activities, season, min_bid=settings.content_min_faab_bid)
    _enrich_managers(moments, _teams(client))
    store = store or Store()
    fresh = _emit(moments, store, notifier, per_scan, generate)
    log.info("Activity cycle: %d activities, %d moments, %d new.",
             len(activities), len(moments), len(fresh))
    return fresh
