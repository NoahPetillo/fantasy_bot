"""Draft engine: snake order, survival math, state, recommender hygiene."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fantasy.draft.adp import survival
from fantasy.draft.recommend import recommend
from fantasy.draft.state import DraftState
from fantasy.league_settings import LeagueSettings, RosterRequirements

LEAGUE = LeagueSettings(
    team_count=4,
    roster=RosterRequirements(slots={"QB": 1, "RB": 1, "WR": 1, "FLEX": 1, "BE": 2}),
)


def _board():
    rows = []
    for i in range(40):
        pos = ["QB", "RB", "WR", "TE"][i % 4]
        rows.append({"player_id": f"p{i}", "player_display_name": f"P{i}", "position": pos,
                     "proj": 200 - i * 4, "vor": 100 - i * 3, "adp": i + 1, "sd": 5.0})
    return pd.DataFrame(rows)


def _state(picks=None):
    return DraftState(LEAGUE, pick_order=[1, 2, 3, 4], rounds=6, board=_board(),
                      my_team_id=1, picks=picks or [])


def test_snake_order_and_my_picks():
    s = _state()
    # Round 1 forward [1,2,3,4]; round 2 reversed [4,3,2,1]; round 3 forward...
    assert [s.team_on_clock(p) for p in range(1, 9)] == [1, 2, 3, 4, 4, 3, 2, 1]
    # Team 1 picks at overall 1, 8, 9, 16, 17, 24.
    assert s.my_pick_numbers()[:4] == [1, 8, 9, 16]
    assert s.my_next_pick(1) == 8


def test_survival_monotonic_and_bounded():
    # Further from ADP -> lower survival; all within [0,1].
    s_near = survival(adp=10, sd=4, next_pick=8)
    s_far = survival(adp=10, sd=4, next_pick=15)
    assert 0.0 <= s_far <= s_near <= 1.0
    # Conditioning on having survived to current raises survival vs unconditional.
    cond = survival(adp=10, sd=4, next_pick=20, current_pick=15)
    uncond = survival(adp=10, sd=4, next_pick=20)
    assert cond >= uncond


def test_state_taken_available_record():
    s = _state(picks=[(1, 2, "p0"), (2, 3, "p1")])
    assert s.taken() == {"p0", "p1"}
    assert "p0" not in set(s.available()["player_id"])
    assert s.current_overall == 3
    s.record_pick(3, 4, "p2")
    assert "p2" in s.taken()


def test_recommend_only_picks_available_and_distinct():
    s = _state(picks=[(1, 2, "p0"), (2, 3, "p1"), (3, 4, "p2")])
    recs = recommend(s, top=5)
    ids = [r.player_id for r in recs]
    assert ids, "expected recommendations"
    assert len(ids) == len(set(ids))                       # no duplicates
    assert all(pid not in s.taken() for pid in ids)        # never a taken player
    assert all(0.0 <= r.survival <= 1.0 for r in recs)
