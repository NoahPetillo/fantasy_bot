"""Trust scoring: base prior (expert tier) × calibration.

v1 uses the manual base_trust priors. The calibration hook is where per-expert
Beta(hits, misses) and the FantasyPros accuracy leaderboard plug in (Phase B) to
auto-tune reliability from outcomes; for now it returns 1.0.
"""

from __future__ import annotations

from fantasy.news.experts.models import ExpertSignal


def calibration(expert_handle: str) -> float:
    # Phase B: blend FantasyPros leaderboard rank + per-expert hit/miss Beta.
    return 1.0


def trust(signal: ExpertSignal) -> float:
    base = signal.base_trust * calibration(signal.expert_handle)
    if signal.is_hedge:
        base *= 0.5
    return max(0.0, min(base, 1.0))
