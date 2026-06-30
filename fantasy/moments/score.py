"""Rank moments by spiciness and pick the few worth posting.

The detector already attaches a 0–100 ``spice`` per moment. Selection sorts by
spice and takes the top N, but with a soft per-matchup cap so we don't post three
different angles on the same game while ignoring the rest of the league.
"""

from __future__ import annotations

from fantasy.config import settings
from fantasy.moments.models import MATCHUP_TYPES, Moment


def spiciness(moment: Moment) -> float:
    """The moment's ranking score (already computed by the detector)."""
    return moment.spice


def rank_and_select(moments: list[Moment], n: int | None = None,
                    max_per_matchup: int = 1) -> list[Moment]:
    """Top-N moments by spice, limiting how many come from one head-to-head.

    Non-matchup moments (superlatives, boom/bust) are never capped against each
    other — only matchup angles on the *same* game compete for the per-matchup slot.
    """
    n = settings.content_moments_per_week if n is None else n
    ranked = sorted(moments, key=lambda m: m.spice, reverse=True)
    seen_matchups: dict[str, int] = {}
    chosen: list[Moment] = []
    for m in ranked:
        if len(chosen) >= n:
            break
        if m.type in MATCHUP_TYPES:
            used = seen_matchups.get(m.dedup_key, 0)
            if used >= max_per_matchup:
                continue
            seen_matchups[m.dedup_key] = used + 1
        chosen.append(m)
    return chosen
