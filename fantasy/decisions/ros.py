"""Rest-of-season scaling — how many games each player actually has left.

"proj × remaining_weeks" over-values players whose bye is still ahead: with a
17-week fantasy season everyone plays the same total, but from TODAY a player
with a future bye has one fewer game to score than one whose bye already passed.
The gap matters most exactly when waiver/trade decisions are made (weeks 4-13).
"""

from __future__ import annotations

import logging

import pandas as pd

from fantasy.league_settings import LeagueSettings

log = logging.getLogger(__name__)


def games_left_by_team(season: int, week: int, remaining_weeks: int) -> dict[str, int]:
    """team -> games the team still plays in the fantasy season window."""
    from fantasy.data.nfl import team_bye_weeks

    try:
        byes = team_bye_weeks(season)
    except Exception as e:  # noqa: BLE001 — schedule fetch is best-effort
        log.warning("Bye weeks unavailable (%s); ROS ignores byes.", e)
        byes = {}
    return {
        team: remaining_weeks - (1 if week <= bye < week + remaining_weeks else 0)
        for team, bye in byes.items()
    }


def ros_maps(
    board: pd.DataFrame, league: LeagueSettings, season: int, week: int, remaining_weeks: int
) -> tuple[dict[str, float], dict[str, float]]:
    """(ros_points, ros_vor) per player_id, bye-aware.

    Falls back to a flat ``remaining_weeks`` multiplier for players whose team
    isn't in the schedule (or when the schedule is unavailable).
    """
    games = games_left_by_team(season, week, remaining_weeks)
    teams = board["team"] if "team" in board.columns else pd.Series("", index=board.index)
    mult = [games.get(t, remaining_weeks) for t in teams]
    ros = dict(zip(board["player_id"], board["proj"].astype(float) * mult))
    ros_vor = dict(zip(board["player_id"], board["vor"].astype(float) * mult))
    return ros, ros_vor
