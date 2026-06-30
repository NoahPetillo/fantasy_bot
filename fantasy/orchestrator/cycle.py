"""One decision cycle: project → build snapshot → generate → persist → notify.

This is the unit the scheduler runs on a cadence. In ``advise`` mode (Phase 2) it
only ever reads + recommends + notifies; nothing is written back to ESPN. New
proposals (by idempotency key) are persisted and pushed to the notifier; repeats
are silently skipped so the user isn't pinged twice for the same advice.
"""

from __future__ import annotations

import logging

import pandas as pd

from fantasy.config import settings
from fantasy.decisions.startsit import recommend_lineup
from fantasy.decisions.trades import recommend_trades
from fantasy.decisions.waivers import recommend_waivers
from fantasy.league_settings import LeagueSettings
from fantasy.league_state import LeagueSnapshot, build_dryrun_snapshot
from fantasy.notify.base import Notifier, get_notifier
from fantasy.orchestrator.models import Proposal, ProposalKind
from fantasy.orchestrator.store import Store
from fantasy.projections.service import ProjectionService

log = logging.getLogger(__name__)

DEFAULT_KINDS = ("start_sit", "waiver", "trade")
_ROSTER_EVENTS = {"injury_out", "ir", "injury_doubtful", "injury_return", "usage_change", "depth_change"}


def order_for_notification(generated: list[Proposal]) -> list[Proposal]:
    """Order proposals for notification. Trades are this manager's biggest edge
    (per the decision audit), so when ``prioritize_trades`` is on they're flagged
    ``priority`` and floated to the front, ahead of waivers/start-sit; within each
    group, higher value first. Otherwise it's plain value-descending."""
    if settings.prioritize_trades:
        for p in generated:
            if p.kind == ProposalKind.trade:
                p.payload["priority"] = True

    def key(p: Proposal):
        trade_first = settings.prioritize_trades and p.kind == ProposalKind.trade
        return (0 if trade_first else 1, -p.value)

    return sorted(generated, key=key)


def fetch_expert_signals() -> list:
    """Best-effort fused expert signals (corroboration-gated). [] on any failure."""
    if not settings.enable_expert_signals:
        return []
    try:
        from fantasy.news.experts import collect_signals
        return collect_signals()
    except Exception as e:  # noqa: BLE001
        log.warning("expert signal collection failed: %s", e)
        return []


def _expert_alerts(snap: LeagueSnapshot, fused: list) -> list[Proposal]:
    roster, fas = set(snap.my_roster()), set(snap.free_agents)
    out = []
    for f in fused:
        mine = f.player_id in roster and f.event_type.value in _ROSTER_EVENTS
        target = f.corroborated and f.direction > 0 and f.player_id in fas
        if not (mine or target):
            continue
        where = "YOUR PLAYER" if mine else "Waiver target"
        out.append(Proposal(
            kind=ProposalKind.alert, season=snap.season, week=snap.week, team_id=snap.my_team_id,
            title=f"📡 {where}: {f.player_name} — {f.event_type.value.replace('_', ' ')}",
            detail=f"{f.rationale}. Experts: {', '.join(f.experts[:4])}.",
            value=round(f.fused_confidence * f.trust_weight * 100, 1), confidence=f.fused_confidence,
            payload={"key_fields": {"sig": f"{f.player_id}:{f.event_type.value}"},
                     "player_id": f.player_id, "corroborated": f.corroborated},
        ))
    return out


def run_cycle(
    service: ProjectionService,
    league: LeagueSettings,
    season: int,
    week: int,
    snapshot: LeagueSnapshot | None = None,
    store: Store | None = None,
    notifier: Notifier | None = None,
    kinds: tuple[str, ...] = DEFAULT_KINDS,
    notify: bool = True,
    weekly: pd.DataFrame | None = None,
    espn_proj: dict[str, float] | None = None,
    fused_signals: list | None = None,
) -> list[Proposal]:
    if fused_signals is None:
        fused_signals = fetch_expert_signals()
    board = service.project(season, week, weekly=weekly, espn_proj=espn_proj,
                            fused_signals=fused_signals)
    if board.empty:
        log.warning("No projections for %s wk %s — nothing to do.", season, week)
        return []
    if snapshot is None:
        snapshot = build_dryrun_snapshot(board, league, season, week)

    rem = service.remaining_weeks(week)
    from fantasy.news.experts.adjust import priority_boosts
    boosts = priority_boosts(fused_signals) if fused_signals else {}

    generated: list[Proposal] = []
    if "start_sit" in kinds:
        generated += recommend_lineup(snapshot, board, league)
    if "waiver" in kinds:
        generated += recommend_waivers(snapshot, board, league, rem, boosts=boosts)
    if "trade" in kinds:
        generated += recommend_trades(snapshot, board, league, rem)
    generated += _expert_alerts(snapshot, fused_signals)

    store = store or Store()
    notifier = notifier if notifier is not None else (get_notifier() if notify else None)

    fresh: list[Proposal] = []
    for p in order_for_notification(generated):
        if store.add(p):  # True => new (not a duplicate idempotency key)
            if notifier is not None:
                ref = notifier.notify(p)
                if ref:
                    store.set_status(p.id, p.status, ref)
            fresh.append(p)
    log.info("Cycle %s wk%s: %d generated, %d new.", season, week, len(generated), len(fresh))
    return fresh
