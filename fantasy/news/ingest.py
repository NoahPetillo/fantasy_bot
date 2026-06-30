"""News ingestion pipeline: gather → dedupe → enrich → alert.

Pulls Sleeper trending + injuries and ESPN news, normalizes to gsis ids, dedupes,
optionally LLM-enriches the high-impact items, then raises ``alert`` proposals for
events that touch the user's roster (injuries/depth/trades to their players) or
surface a trending waiver target available in their league.
"""

from __future__ import annotations

import logging

from fantasy.league_state import LeagueSnapshot
from fantasy.news import espn_news, extract, sleeper
from fantasy.news.models import EventType, PlayerSignal
from fantasy.orchestrator.models import Proposal, ProposalKind
from fantasy.orchestrator.store import Store

log = logging.getLogger(__name__)

_ROSTER_EVENTS = {EventType.injury_out, EventType.ir, EventType.injury_doubtful,
                  EventType.injury_questionable, EventType.injury_return,
                  EventType.depth_change, EventType.trade}
_ICON = {EventType.injury_out: "🚑", EventType.ir: "🏥", EventType.injury_doubtful: "⚠️",
         EventType.injury_questionable: "❓", EventType.injury_return: "✅",
         EventType.depth_change: "📊", EventType.trade: "🔄", EventType.trending_add: "📈"}


def gather_signals(player_ids: set[str] | None = None, trending_limit: int = 25,
                   news_limit: int = 50, enrich: bool = True) -> list[PlayerSignal]:
    signals: list[PlayerSignal] = []
    for fn in (lambda: sleeper.trending_signals(trending_limit),
               lambda: sleeper.injury_signals(player_ids),
               lambda: espn_news.news_signals(news_limit)):
        try:
            signals += fn()
        except Exception as e:  # noqa: BLE001
            log.warning("News source failed: %s", e)
    # dedupe by (player, event, ts)
    seen, deduped = set(), []
    for s in signals:
        if s.key() not in seen:
            seen.add(s.key())
            deduped.append(s)
    if enrich:
        deduped = extract.enrich_signals(deduped)
    return deduped


def make_alerts(snap: LeagueSnapshot, signals: list[PlayerSignal]) -> list[Proposal]:
    roster = set(snap.my_roster())
    fas = set(snap.free_agents)
    props: list[Proposal] = []
    for s in signals:
        on_my_team = s.player_id in roster and s.event_type in _ROSTER_EVENTS
        trending_target = s.event_type == EventType.trending_add and s.player_id in fas
        if not (on_my_team or trending_target):
            continue
        icon = _ICON.get(s.event_type, "📰")
        if on_my_team:
            title = f"{icon} {s.event_type.value.replace('_', ' ').title()}: {s.player_name}"
            detail = f"YOUR PLAYER. {s.summary} (via {s.source})."
        else:
            title = f"{icon} Trending waiver target: {s.player_name} ({s.position or '?'})"
            detail = f"Available in your league. {s.summary}."
        if s.beneficiaries:
            detail += f"\nLikely beneficiaries: {', '.join(s.beneficiaries[:4])}."
        props.append(Proposal(
            kind=ProposalKind.alert, season=snap.season, week=snap.week, team_id=snap.my_team_id,
            title=title, detail=detail, value=round(s.severity * 100, 1), confidence=s.confidence,
            payload={"key_fields": {"sig": s.key()}, "player_id": s.player_id,
                     "event": s.event_type.value, "beneficiaries": s.beneficiaries},
        ))
    return props


def news_cycle(snap: LeagueSnapshot, store: Store | None = None, notifier=None,
               enrich: bool = True) -> list[Proposal]:
    store = store or Store()
    signals = gather_signals(player_ids=set(snap.my_roster()) | set(snap.free_agents), enrich=enrich)
    alerts = make_alerts(snap, signals)
    fresh = []
    for p in sorted(alerts, key=lambda x: x.value, reverse=True):
        if store.add(p):
            if notifier is not None:
                ref = notifier.notify(p)
                if ref:
                    store.set_status(p.id, p.status, ref)
            fresh.append(p)
    log.info("News cycle: %d signals, %d alerts, %d new.", len(signals), len(alerts), len(fresh))
    return fresh
