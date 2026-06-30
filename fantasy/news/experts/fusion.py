"""Fuse expert signals into trust-weighted, corroboration-gated consensus.

Per (player, event-family): drop sarcasm/hypotheticals, decay by recency, combine
by trust, then gate. A fused signal may MOVE a decision only if it's corroborated:
≥2 independent OUTLETS agree, OR a high-trust breaker on an injury. Otherwise it's
alert-only (surfaced to the human, never auto-applied).
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from fantasy.news.experts.models import ExpertSignal, FusedSignal
from fantasy.news.experts.trust import trust
from fantasy.news.models import SEVERITY, EventType

_FAMILY = {
    EventType.injury_out: "injury", EventType.injury_doubtful: "injury",
    EventType.injury_questionable: "injury", EventType.ir: "injury",
    EventType.injury_return: "injury", EventType.depth_change: "usage",
    EventType.usage_change: "usage", EventType.breakout: "usage",
    EventType.trending_add: "usage", EventType.trending_drop: "usage",
    EventType.buy_low: "value", EventType.sell_high: "value",
}
_HALF_LIFE_H = {"injury": 12.0, "usage": 72.0, "value": 168.0}


def _recency_weight(ts: str | None, family: str) -> float:
    if not ts:
        return 1.0
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        age_h = (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0
        return 0.5 ** (max(age_h, 0) / _HALF_LIFE_H.get(family, 72.0))
    except (ValueError, TypeError):
        return 1.0


def fuse(signals: list[ExpertSignal]) -> list[FusedSignal]:
    groups: dict[tuple[str, str], list[ExpertSignal]] = {}
    for s in signals:
        if s.is_sarcasm or s.is_hypothetical or not s.player_id:
            continue  # hard-zero
        fam = _FAMILY.get(s.event_type, "usage")
        groups.setdefault((s.player_id, fam), []).append(s)

    fused: list[FusedSignal] = []
    for (pid, fam), grp in groups.items():
        num = den = 0.0
        for s in grp:
            w = trust(s) * _recency_weight(s.ts, fam)
            num += w * s.direction
            den += w
        if den <= 0:
            continue
        direction = 1 if num > 0 else (-1 if num < 0 else 0)
        agreeing = [s for s in grp if s.direction == direction or direction == 0]
        outlets = {s.outlet for s in agreeing}
        trust_weight = max(trust(s) for s in agreeing)
        weight_sum = sum(trust(s) * _recency_weight(s.ts, fam) for s in agreeing)
        fused_conf = min(1.0 - math.exp(-weight_sum), 0.99)
        # canonical event = most severe in the group
        event = max(grp, key=lambda s: SEVERITY.get(s.event_type, 0.2)).event_type

        corroborated = len(outlets) >= 2 or (
            fam == "injury" and trust_weight >= 0.9
        )
        fused.append(FusedSignal(
            player_id=pid, player_name=grp[0].player_name, event_type=event,
            direction=direction, fused_confidence=round(fused_conf, 3),
            trust_weight=round(trust_weight, 3), independent_outlets=len(outlets),
            corroborated=corroborated, experts=sorted({s.expert_handle for s in agreeing}),
            rationale=(f"{len(agreeing)} signal(s) from {len(outlets)} outlet(s); "
                       f"{'corroborated' if corroborated else 'single-source (alert-only)'}"),
        ))
    return sorted(fused, key=lambda f: f.fused_confidence * f.trust_weight, reverse=True)
