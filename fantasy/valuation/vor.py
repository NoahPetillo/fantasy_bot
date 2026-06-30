"""Value Over Replacement (VOR / VBD) — parameterized entirely by league settings.

Replacement level is the projected output of the *last startable* player at a
position, where "startable count" derives from the league's roster slots and team
count (dedicated slots + fractional flex/superflex share). VOR makes points
comparable across positions, which is what every downstream decision needs:

- weekly VOR  -> start/sit, FAAB marginal value
- ROS VOR     -> trade value, draft value

No format is hardcoded — it reads team_count + roster from LeagueSettings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fantasy.league_settings import LeagueSettings


def replacement_counts(settings: LeagueSettings) -> dict[str, int]:
    """League-wide count of startable players per position (the replacement rank)."""
    counts: dict[str, int] = {}
    for pos in ["QB", "RB", "WR", "TE", "K", "D/ST"]:
        per_team = settings.roster.starters_at_position(pos)
        counts[pos] = int(round(settings.team_count * per_team))
    return counts


def replacement_baselines(
    proj: pd.DataFrame, settings: LeagueSettings, proj_col: str = "proj"
) -> dict[str, float]:
    """Replacement points per position = robust avg around the last-starter rank.

    Averages ranks [r-1, r, r+1] (1-based) to avoid a single noisy player setting
    the baseline. Positions with too few players fall back to their min projection.
    """
    counts = replacement_counts(settings)
    baselines: dict[str, float] = {}
    for pos, rank in counts.items():
        vals = (
            proj.loc[proj["position"] == pos, proj_col]
            .dropna()
            .sort_values(ascending=False)
            .to_numpy()
        )
        if len(vals) == 0 or rank <= 0:
            baselines[pos] = 0.0
            continue
        lo, hi = max(rank - 2, 0), min(rank + 1, len(vals))
        window = vals[lo:hi] if hi > lo else vals[-1:]
        baselines[pos] = float(np.mean(window))
    return baselines


def compute_vor(
    proj: pd.DataFrame, settings: LeagueSettings, proj_col: str = "proj"
) -> pd.DataFrame:
    """Add ``replacement`` and ``vor`` columns to a frame of [position, proj_col].

    VOR can be negative (below replacement = waiver fodder). Sorted by VOR desc so
    the result doubles as a cross-positional draft/value board.
    """
    out = proj.copy()
    baselines = replacement_baselines(out, settings, proj_col)
    out["replacement"] = out["position"].map(baselines).fillna(0.0)
    out["vor"] = (out[proj_col] - out["replacement"]).round(2)
    return out.sort_values("vor", ascending=False)
