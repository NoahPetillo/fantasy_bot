"""DraftState — the evolving state a recommender reasons over.

Snake order + "my next pick" come from the pick order + round parity formula the
design pass validated against all 192 real picks with zero mismatches. The same
state object is fed by self-play, the real-draft replay, or the live poller — the
recommender doesn't care where picks come from.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from fantasy.league_settings import LeagueSettings


@dataclass
class DraftState:
    league: LeagueSettings
    pick_order: list[int]            # team ids in round-1 order
    rounds: int
    board: pd.DataFrame              # player_id, position, proj, vor, adp, sd, ...
    my_team_id: int
    picks: list[tuple[int, int, str]] = field(default_factory=list)  # (overall, team, gsis)

    @property
    def num_teams(self) -> int:
        return len(self.pick_order)

    @property
    def current_overall(self) -> int:
        return len(self.picks) + 1

    def team_on_clock(self, overall: int) -> int:
        n = self.num_teams
        round0, idx = divmod(overall - 1, n)
        order = self.pick_order if round0 % 2 == 0 else self.pick_order[::-1]
        return order[idx]

    def my_pick_numbers(self) -> list[int]:
        total = self.num_teams * self.rounds
        return [p for p in range(1, total + 1) if self.team_on_clock(p) == self.my_team_id]

    def my_next_pick(self, after: int | None = None) -> int | None:
        after = after if after is not None else self.current_overall - 1
        return next((p for p in self.my_pick_numbers() if p > after), None)

    def picks_until_my_turn(self) -> int:
        nxt = self.my_next_pick(self.current_overall - 1)
        return (nxt - self.current_overall) if nxt else 0

    def taken(self) -> set[str]:
        return {pid for _, _, pid in self.picks}

    def available(self) -> pd.DataFrame:
        return self.board[~self.board["player_id"].isin(self.taken())]

    def my_roster(self) -> list[str]:
        return [pid for _, team, pid in self.picks if team == self.my_team_id]

    def roster_of(self, team_id: int) -> list[str]:
        return [pid for _, team, pid in self.picks if team == team_id]

    def record_pick(self, overall: int, team_id: int, player_id: str) -> None:
        self.picks.append((overall, team_id, player_id))

    def is_my_turn(self) -> bool:
        return self.team_on_clock(self.current_overall) == self.my_team_id

    def is_complete(self) -> bool:
        return len(self.picks) >= self.num_teams * self.rounds
