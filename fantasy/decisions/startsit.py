"""Start/sit recommendations from the optimal lineup."""

from __future__ import annotations

import pandas as pd

from fantasy.decisions.lineup import optimize_lineup
from fantasy.league_settings import LeagueSettings
from fantasy.league_state import LeagueSnapshot
from fantasy.orchestrator.models import Proposal, ProposalKind


def recommend_lineup(
    snap: LeagueSnapshot, board: pd.DataFrame, league: LeagueSettings
) -> list[Proposal]:
    mine = board[board["player_id"].isin(snap.my_roster())].copy()
    if mine.empty:
        return []
    players = [(r.player_id, r.position, float(r.proj)) for r in mine.itertuples(index=False)]
    lineup = optimize_lineup(players, league)

    starters = {pid for pids in lineup.values() for pid in pids}
    name = dict(zip(mine["player_id"], mine["player_display_name"]))
    pos = dict(zip(mine["player_id"], mine["position"]))
    proj = dict(zip(mine["player_id"], mine["proj"]))
    fl = dict(zip(mine["player_id"], mine["floor"]))
    ce = dict(zip(mine["player_id"], mine["ceiling"]))

    start_lines, total = [], 0.0
    for slot, pids in lineup.items():
        for pid in pids:
            total += proj[pid]
            start_lines.append(
                f"  {slot:5s} {name[pid]:22s} {proj[pid]:5.1f}  ({fl[pid]:.0f}-{ce[pid]:.0f})"
            )

    # Close calls: best benched player vs the weakest starter at an overlapping slot.
    bench = mine[~mine["player_id"].isin(starters)].sort_values("proj", ascending=False)
    swaps = []
    for b in bench.head(6).itertuples(index=False):
        weaker = [s for s in starters if proj[s] < b.proj and _same_flex(pos[s], b.position, league)]
        if weaker:
            worst = min(weaker, key=lambda s: proj[s])
            swaps.append(f"  consider {name[b.player_id]} ({b.proj:.1f}) over "
                         f"{name[worst]} ({proj[worst]:.1f})  +{b.proj - proj[worst]:.1f}")

    detail = "Optimal starting lineup:\n" + "\n".join(start_lines)
    if swaps:
        detail += "\n\nClose calls:\n" + "\n".join(swaps)
    return [
        Proposal(
            kind=ProposalKind.start_sit, season=snap.season, week=snap.week,
            team_id=snap.my_team_id, title=f"Week {snap.week} lineup — {total:.1f} proj pts",
            detail=detail, value=round(total, 2),
            payload={"key_fields": {"starters": sorted(starters)},
                     "lineup": {s: p for s, p in lineup.items() if p}},
        )
    ]


def _same_flex(pos_a: str, pos_b: str, league: LeagueSettings) -> bool:
    if pos_a == pos_b:
        return True
    # Both eligible for the same flex slot present in the league.
    from fantasy.espn.stat_ids import FLEX_ELIGIBILITY

    for slot in league.roster.starter_slots:
        elig = FLEX_ELIGIBILITY.get(slot)
        if elig and pos_a in elig and pos_b in elig:
            return True
    return False
