"""Usage redistribution — "the backup spikes when the starter is ruled out".

When our injury ingestion flags a starter OUT, the next man up inherits a chunk of
the vacated workload. Markets and slow projections under-adjust this for a day or
two, so applying it the instant we detect the ruling is a real, free timing edge.

We identify the backup as the **highest-projected other player at the same team +
position** (robust — our own board, not noisy public depth-chart ordering), and
hand them a position-appropriate share of the ruled-out starter's projection.
"""

from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)

_SKILL = {"QB", "RB", "WR", "TE"}
# Share of a ruled-out starter's projection handed to the next man up, by position.
_REDISTRIB = {"RB": 0.55, "WR": 0.35, "TE": 0.45, "QB": 0.85}


def vacated_boosts(out_ids: list[str], board: pd.DataFrame,
                   factor_scale: float = 1.0) -> dict[str, float]:
    """gsis -> projection boost for backups of ruled-out starters.

    ``board`` needs columns player_id, team, position, proj. For each OUT player,
    the highest-projected remaining same-team/position player inherits
    ``_REDISTRIB[pos]`` of the starter's projection (capped at the vacated value).
    """
    if not out_ids or board.empty or "team" not in board.columns:
        return {}
    out_set = set(out_ids)
    b = board.set_index("player_id")
    boosts: dict[str, float] = {}
    for out in out_ids:
        if out not in b.index:
            continue
        row = b.loc[out]
        row = row.iloc[0] if isinstance(row, pd.DataFrame) else row
        team, pos, vacated = row.get("team"), row.get("position"), float(row.get("proj", 0.0))
        if pos not in _SKILL or vacated <= 0 or not isinstance(team, str):
            continue
        cands = board[(board["team"] == team) & (board["position"] == pos)
                      & (~board["player_id"].isin(out_set))].sort_values("proj", ascending=False)
        if cands.empty:
            continue
        backup = cands.iloc[0]["player_id"]
        boost = min(_REDISTRIB.get(pos, 0.4) * factor_scale * vacated, vacated)
        boosts[backup] = boosts.get(backup, 0.0) + round(boost, 2)
    return boosts
