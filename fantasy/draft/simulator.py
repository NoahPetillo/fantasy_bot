"""Self-play draft simulator.

Runs N-team snake drafts where opponents follow ADP-with-noise and one seat uses
our recommender. Rosters are then scored by ACTUAL that-season points (the best
season-long starting lineup), so we measure real draft skill, not hindsight.
DST/K aren't in our scoring data, so realized value reflects offensive starters
(an equal limitation across all agents).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fantasy.data.nfl import season_totals
from fantasy.decisions.lineup import greedy_lineup
from fantasy.draft.recommend import _DEPTH, best_pick
from fantasy.draft.state import DraftState
from fantasy.league_settings import LeagueSettings
from fantasy.valuation.scoring import ScoringEngine


class OurAgent:
    name = "ours"

    def __init__(self, strategy: str = "none"):
        self.strategy = strategy

    def pick(self, state: DraftState, rng) -> str | None:
        return best_pick(state, self.strategy)


class AdpAgent:
    """Drafts near ADP with noise, with light positional sanity (no absurd rosters)."""

    name = "adp"

    def pick(self, state: DraftState, rng) -> str | None:
        avail = state.available()
        if avail.empty:
            return None
        board_pos = state.board.set_index("player_id")["position"]
        have = pd.Series([board_pos.get(p) for p in state.my_roster()]).value_counts().to_dict()

        def over(pos):
            import math
            cap = math.ceil(state.league.roster.starters_at_position(pos)) + _DEPTH.get(pos, 1) + 1
            return have.get(pos, 0) >= cap

        ok = avail[~avail["position"].map(over)]
        pool = ok if not ok.empty else avail
        noise = rng.normal(0, np.maximum(pool["sd"].to_numpy(float), 1.0))
        key = pool["adp"].to_numpy(float) + noise
        return pool.iloc[int(key.argmin())]["player_id"]


class RandomAgent:
    name = "random"

    def pick(self, state: DraftState, rng) -> str | None:
        avail = state.available()
        return None if avail.empty else avail.iloc[int(rng.integers(len(avail)))]["player_id"]


def run_draft(league: LeagueSettings, board: pd.DataFrame, agents: dict, pick_order: list[int],
              rounds: int, seed: int = 0) -> DraftState:
    state = DraftState(league, list(pick_order), rounds, board, my_team_id=pick_order[0])
    rng = np.random.default_rng(seed)
    while not state.is_complete():
        overall = state.current_overall
        team = state.team_on_clock(overall)
        state.my_team_id = team  # give the agent this team's perspective
        pid = agents[team].pick(state, rng)
        if pid is None:
            break
        state.record_pick(overall, team, pid)
    return state


def score_rosters(state: DraftState, season: int, league: LeagueSettings) -> dict[int, float]:
    """Best season-long starting-lineup actual points, per team."""
    actual = season_totals([season], ScoringEngine(league))
    pts = dict(zip(actual["player_id"], actual["pts"]))
    board_pos = state.board.set_index("player_id")["position"].to_dict()
    out = {}
    for team in state.pick_order:
        roster = state.roster_of(team)
        proj = {p: pts.get(p, 0.0) for p in roster}
        out[team] = greedy_lineup(proj, board_pos, roster, league)[0]
    return out
