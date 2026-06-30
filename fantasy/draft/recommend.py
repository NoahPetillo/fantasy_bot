"""Pick recommendation — survival-weighted VONA (the cheap, default tier).

For each available player:
    score = VOR * need + (1 - survival) * dropoff
- VOR            : cross-positional value over replacement (draft board)
- need           : roster-need multiplier (a 3rd QB is worth little)
- survival       : P(player lasts to my NEXT pick) from the ADP model
- dropoff        : value gap to the next-best available at the same position

So the engine grabs a high-value player precisely when he won't survive and the
drop-off behind him is steep — it reads positional runs and avoids reaching.
K/DST are gated to the last rounds; soft strategy priors bias the early rounds.
The core is vectorized so the same logic drives live recs and fast self-play.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from pydantic import BaseModel
from scipy.stats import norm

from fantasy.draft.state import DraftState
from fantasy.league_settings import LeagueSettings

_DEPTH = {"QB": 1, "TE": 1, "RB": 4, "WR": 4, "K": 0, "D/ST": 0, "DST": 0}
_STRATEGY = {
    "none": {},
    "zero_rb": {"RB": 0.7, "WR": 1.15, "TE": 1.1},
    "hero_rb": {"RB": 0.9, "WR": 1.1},
    "robust_rb": {"RB": 1.2, "WR": 0.95},
}


class PickRec(BaseModel):
    player_id: str
    name: str
    position: str
    vor: float
    adp: float
    survival: float
    score: float
    reason: str


def _need(have: int, pos: str, league: LeagueSettings) -> float:
    starters = math.ceil(league.roster.starters_at_position(pos))
    if have < starters:
        return 1.0
    if have < starters + _DEPTH.get(pos, 1):
        return 0.55
    return 0.15


def _survival_vec(adp: np.ndarray, sd: np.ndarray, next_pick: int, cur: int) -> np.ndarray:
    sigma = np.maximum(np.maximum(sd, 0.5 * np.sqrt(np.maximum(adp, 0))), 1.0)
    p_next = 1.0 - norm.cdf(next_pick, loc=adp, scale=sigma)
    if cur and cur > 0:
        p_cur = 1.0 - norm.cdf(cur, loc=adp, scale=sigma)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(p_cur > 1e-9, p_next / p_cur, 1.0)
        return np.clip(ratio, 0.0, 1.0)
    return np.clip(p_next, 0.0, 1.0)


def score_available(state: DraftState, strategy: str = "none") -> pd.DataFrame:
    """Available players with a 'score' (and survival) column — vectorized."""
    avail = state.available().copy()
    if avail.empty:
        return avail
    board_pos = state.board.set_index("player_id")["position"]
    pos_counts = (
        pd.Series([board_pos.get(p) for p in state.my_roster()]).value_counts().to_dict()
    )

    cur = state.current_overall
    next_pick = state.my_next_pick(cur - 1) or (cur + 1)
    rounds_left = state.rounds - (cur - 1) // state.num_teams
    prior = _STRATEGY.get(strategy, {})
    early = rounds_left >= state.rounds - 5

    avail["survival"] = _survival_vec(avail["adp"].to_numpy(float), avail["sd"].to_numpy(float),
                                      next_pick, cur)
    # drop-off: VOR minus the next-best available VOR at the same position.
    nb = avail.sort_values("vor", ascending=False).groupby("position")["vor"].shift(-1)
    avail["dropoff"] = (avail["vor"] - nb.reindex(avail.index)).clip(lower=0).fillna(0)
    need = avail["position"].map(lambda p: _need(pos_counts.get(p, 0), p, state.league))
    gate = avail["position"].map(
        lambda p: 1.0 if p not in ("K", "D/ST", "DST") else (1.0 if rounds_left <= 2 else 0.03))
    mult = avail["position"].map(lambda p: prior.get(p, 1.0)) if early else 1.0
    avail["score"] = (avail["vor"] * need + (1 - avail["survival"]) * avail["dropoff"]) * gate * mult
    return avail.sort_values("score", ascending=False)


def recommend(state: DraftState, strategy: str = "none", top: int = 8) -> list[PickRec]:
    scored = score_available(state, strategy)
    if scored.empty:
        return []
    next_pick = state.my_next_pick(state.current_overall - 1) or (state.current_overall + 1)
    out = []
    for r in scored.head(top).itertuples():
        if r.survival < 0.35 and r.dropoff > 3:
            reason = f"likely gone by your next pick (P{next_pick}); steep drop-off after him"
        elif r.score == scored["score"].max():
            reason = f"best available value (VOR {r.vor:+.0f}, ~{r.survival*100:.0f}% to survive)"
        else:
            reason = f"VOR {r.vor:+.0f}, ~{r.survival*100:.0f}% to last to your next pick"
        out.append(PickRec(player_id=r.player_id, name=r.player_display_name, position=r.position,
                           vor=round(r.vor, 1), adp=round(r.adp, 1),
                           survival=round(r.survival, 2), score=round(r.score, 2), reason=reason))
    return out


def best_pick(state: DraftState, strategy: str = "none") -> str | None:
    scored = score_available(state, strategy)
    return None if scored.empty else scored.iloc[0]["player_id"]
