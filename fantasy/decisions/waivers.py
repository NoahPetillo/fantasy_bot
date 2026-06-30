"""Waiver / FAAB recommendations.

A free-agent add is valued by how much it raises your overall ROSTER value, where
roster value = your starting-lineup points (ROS) + a discounted credit for BENCH
depth (each bench player's positive ROS VOR, weighted, since bench points are
contingent — bye fills, injury insurance, upside stashes). So a pickup is worth it
if it either cracks your lineup OR is simply better than the bench player you'd
drop. We drop your weakest bench player and report "pick up X, drop Y".

This avoids the earlier trap of over-valuing a pickup just because the nominal
drop had deeply negative VOR.
"""

from __future__ import annotations

import pandas as pd

from fantasy.decisions.faab import suggest_bid
from fantasy.decisions.lineup import greedy_lineup
from fantasy.league_settings import LeagueSettings, WaiverType
from fantasy.league_state import LeagueSnapshot
from fantasy.orchestrator.models import Proposal, ProposalKind

# How much a bench player's ROS VOR counts vs a starter's points (bench is contingent).
BENCH_WEIGHT = 0.25


def _roster_value(roster, ros, ros_vor, pos, league) -> tuple[float, set[str]]:
    """Lineup points (ROS) + discounted bench-depth credit. Returns (value, starters)."""
    lineup_pts, starters = greedy_lineup(ros, pos, roster, league)
    bench_credit = BENCH_WEIGHT * sum(max(ros_vor.get(p, 0.0), 0.0)
                                      for p in roster if p not in starters)
    return lineup_pts + bench_credit, starters


def recommend_waivers(
    snap: LeagueSnapshot, board: pd.DataFrame, league: LeagueSettings,
    remaining_weeks: int, top_k: int = 5, min_gain: float = 1.0, scan: int = 60,
    boosts: dict[str, float] | None = None,
) -> list[Proposal]:
    b = board.set_index("player_id")
    my = [p for p in snap.my_roster() if p in b.index]
    fas = [p for p in snap.free_agents if p in b.index]
    if not my or not fas:
        return []

    ros = {pid: float(b.loc[pid, "proj"]) * remaining_weeks for pid in b.index}
    ros_vor = {pid: float(b.loc[pid, "vor"]) * remaining_weeks for pid in b.index}
    pos = {pid: b.loc[pid, "position"] for pid in b.index}

    base_val, starters = _roster_value(my, ros, ros_vor, pos, league)
    bench = [p for p in my if p not in starters]
    # Drop the weakest bench player (never auto-drop a starter).
    pool = bench or my
    drop = min(pool, key=lambda p: ros_vor.get(p, 0.0))
    drow = b.loc[drop]

    budget = league.faab_budget
    remaining = snap.faab_remaining.get(snap.my_team_id, budget)
    cand = b.loc[fas].sort_values("vor", ascending=False).head(scan)

    from fantasy.config import settings
    use_boosts = bool(boosts) and settings.expert_adjust_decisions

    props: list[Proposal] = []
    for fa in cand.itertuples():
        add = fa.Index
        if add == drop:
            continue
        new_val, _ = _roster_value([p for p in my if p != drop] + [add], ros, ros_vor, pos, league)
        gain = new_val - base_val
        boost = boosts.get(add, 1.0) if use_boosts else 1.0
        gain *= boost  # experts flagging this add raise its waiver priority (capped ≤1.5×)
        if gain < min_gain:
            continue
        cracks = "starting lineup" if _cracks_lineup(my, drop, add, ros, pos, league) else "bench depth"
        bid = (suggest_bid(gain, budget, remaining)
               if league.waiver_type == WaiverType.faab else 0)
        bid_str = f"  •  bid ${bid}" if league.waiver_type == WaiverType.faab else "  •  waiver claim"
        boost_note = f" Expert-boosted ×{boost:.2f}." if boost > 1.0 else ""
        props.append(Proposal(
            kind=ProposalKind.waiver, season=snap.season, week=snap.week, team_id=snap.my_team_id,
            title=f"Add {fa.player_display_name} ({fa.position}) / drop {drow['player_display_name']}{bid_str}",
            detail=(f"Pick up {fa.player_display_name} ({fa.position}, {fa.proj:.1f} proj, "
                    f"VOR {float(fa.vor):+.1f}) and drop {drow['player_display_name']} "
                    f"({drow['position']}, VOR {float(drow['vor']):+.1f}).\n"
                    f"Upgrades your {cracks} by ~{gain:.0f} ROS pts.{boost_note}"),
            value=round(gain, 2), confidence=min(0.5 + gain / 40, 0.95),
            payload={"key_fields": {"add": add, "drop": drop}, "add": add, "drop": drop,
                     "faab_bid": bid, "upgrades": cracks, "expert_boost": round(boost, 2)},
        ))
        if len(props) >= top_k:
            break
    return props


def _cracks_lineup(roster, drop, add, ros, pos, league) -> bool:
    _, starters = greedy_lineup(ros, pos, [p for p in roster if p != drop] + [add], league)
    return add in starters
