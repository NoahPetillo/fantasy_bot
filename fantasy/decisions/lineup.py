"""Lineup optimization — assign rostered players to starting slots.

Two entry points:
- ``optimize_lineup``: exact ILP (PuLP) for the official start/sit recommendation.
- ``lineup_value``: fast greedy valuation used inside trade search (called many
  times, where exactness matters less than speed).

Both are parameterized by the league's roster slots + flex eligibility.
"""

from __future__ import annotations

import pulp

from fantasy.espn.stat_ids import FLEX_ELIGIBILITY
from fantasy.league_settings import LeagueSettings


def _eligible(position: str, slot: str) -> bool:
    if slot in FLEX_ELIGIBILITY:
        return position in FLEX_ELIGIBILITY[slot]
    return slot == position


def optimize_lineup(
    players: list[tuple[str, str, float]], league: LeagueSettings
) -> dict[str, list[str]]:
    """Exact optimal lineup. ``players`` = [(player_id, position, proj)].

    Returns slot -> [player_id]. Maximizes total projected points subject to slot
    counts and eligibility; each player starts in at most one slot.
    """
    slots = league.roster.starter_slots
    prob = pulp.LpProblem("lineup", pulp.LpMaximize)
    # x[(pid, slot)] = 1 if player pid starts in slot
    x = {}
    for pid, pos, proj in players:
        for slot, count in slots.items():
            if count > 0 and _eligible(pos, slot):
                x[(pid, slot)] = pulp.LpVariable(f"x_{pid}_{slot}", cat="Binary")

    proj_map = {pid: proj for pid, _, proj in players}
    prob += pulp.lpSum(var * proj_map[pid] for (pid, slot), var in x.items())

    # each slot filled exactly its count (or as many as eligible players allow)
    for slot, count in slots.items():
        if count > 0:
            vars_in_slot = [v for (pid, s), v in x.items() if s == slot]
            if vars_in_slot:
                prob += pulp.lpSum(vars_in_slot) <= count
    # each player starts at most once
    for pid, _, _ in players:
        pv = [v for (p, s), v in x.items() if p == pid]
        if pv:
            prob += pulp.lpSum(pv) <= 1

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    out: dict[str, list[str]] = {s: [] for s in slots}
    for (pid, slot), var in x.items():
        if var.value() and var.value() > 0.5:
            out[slot].append(pid)
    return out


def greedy_lineup(
    proj_by_player: dict[str, float], pos_by_player: dict[str, str],
    player_ids: list[str], league: LeagueSettings,
) -> tuple[float, set[str]]:
    """Fast greedy best legal lineup; returns (total points, set of starters)."""
    slots = league.roster.starter_slots
    avail = sorted(
        [(pid, pos_by_player.get(pid, ""), proj_by_player.get(pid, 0.0)) for pid in player_ids],
        key=lambda t: t[2], reverse=True,
    )
    used: set[str] = set()
    total = 0.0
    # Fill dedicated slots first, then flexes (most-restrictive first).
    ordered_slots = sorted(slots.items(), key=lambda kv: (kv[0] in FLEX_ELIGIBILITY,
                                                          len(FLEX_ELIGIBILITY.get(kv[0], {1}))))
    for slot, count in ordered_slots:
        filled = 0
        for pid, pos, proj in avail:
            if filled >= count:
                break
            if pid in used or not _eligible(pos, slot):
                continue
            used.add(pid)
            total += proj
            filled += 1
    return round(total, 2), used


def lineup_value(
    proj_by_player: dict[str, float], pos_by_player: dict[str, str],
    player_ids: list[str], league: LeagueSettings,
) -> float:
    """Fast greedy total projected points of the best legal lineup."""
    return greedy_lineup(proj_by_player, pos_by_player, player_ids, league)[0]
