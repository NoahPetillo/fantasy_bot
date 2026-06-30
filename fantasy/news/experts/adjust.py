"""Turn fused signals into bounded adjustments to the decision stack.

- projection_delta: a capped nudge to a player's projected points (applied between
  service.overlay_espn and _finish_board so floor/ceiling + VOR recompute).
- priority_boost: a capped multiplier (>=1.0) on a waiver add's value / FAAB bid.

Only CORROBORATED signals adjust; single-source signals are alert-only. All
adjustments are hard-capped so a hot take can never dominate the model.
"""

from __future__ import annotations

from fantasy.news.experts.models import FusedSignal
from fantasy.news.models import SEVERITY

# Caps: a fully-trusted, max-severity signal can move a projection by at most this
# fraction, and boost a waiver bid by at most this much.
MAX_DELTA_FRAC = 0.15
MAX_BOOST = 1.5


def projection_deltas(fused: list[FusedSignal], proj: dict[str, float]) -> dict[str, float]:
    """gsis -> additive points delta (corroborated signals only, capped per player)."""
    out: dict[str, float] = {}
    for f in fused:
        if not f.corroborated or f.player_id not in proj:
            continue
        strength = f.direction * f.fused_confidence * f.trust_weight * SEVERITY.get(f.event_type, 0.2)
        cap = MAX_DELTA_FRAC * max(proj[f.player_id], 1.0)
        out[f.player_id] = max(-cap, min(strength * proj[f.player_id], cap))
    return out


def priority_boosts(fused: list[FusedSignal]) -> dict[str, float]:
    """gsis -> waiver/FAAB multiplier (>=1.0) for positive corroborated add signals."""
    out: dict[str, float] = {}
    for f in fused:
        if not f.corroborated or f.direction <= 0:
            continue
        boost = 1.0 + min(MAX_BOOST - 1.0,
                          f.fused_confidence * f.trust_weight * SEVERITY.get(f.event_type, 0.2))
        out[f.player_id] = max(out.get(f.player_id, 1.0), round(boost, 3))
    return out


def alerts(fused: list[FusedSignal]) -> list[FusedSignal]:
    """Single-source / uncorroborated signals worth surfacing to the human only."""
    return [f for f in fused if not f.corroborated and f.fused_confidence >= 0.4]
