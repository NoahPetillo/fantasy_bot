"""Multi-source consensus — the durable edge over any single projection source.

Equal-weight mean across whatever sources are available per player (research:
simple averaging beats learned weighting and beats individual sources, incl. ESPN,
year over year; the gain is from method diversity, not source count). Players
covered by more sources get a small confidence signal via the source count.
"""

from __future__ import annotations

import numpy as np


def consensus(sources: dict[str, dict[str, float]], min_sources: int = 1
              ) -> tuple[dict[str, float], dict[str, int]]:
    """Average per-player projections across sources.

    ``sources`` = {source_name: {gsis_id: points}}. Returns (mean_by_player,
    n_sources_by_player). A player is included if covered by >= ``min_sources``.
    """
    all_ids: set[str] = set().union(*[set(s) for s in sources.values()]) if sources else set()
    means: dict[str, float] = {}
    counts: dict[str, int] = {}
    for pid in all_ids:
        vals = [s[pid] for s in sources.values() if pid in s and s[pid] is not None]
        if len(vals) >= min_sources:
            means[pid] = float(np.mean(vals))
            counts[pid] = len(vals)
    return means, counts
