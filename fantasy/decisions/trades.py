"""Trade discovery — surface win-win 1-for-1 offers.

For each opponent we evaluate swapping one of my players for one of theirs and
measure the rest-of-season *starting-lineup* gain for BOTH teams (greedy lineup
value on ROS-scaled projections). A trade is a candidate only if both sides
improve. We rank by ``my_gain * P(accept)``, where acceptance rises with the
opponent's own lineup gain — i.e., offers that genuinely help them too.

This is the engine behind the proactive "trades you should propose" notifications.
Market-value fairness (FantasyCalc) and a learned per-manager acceptance model are
later refinements; ROS lineup gain is the honest v1 signal.
"""

from __future__ import annotations

import math

import pandas as pd

from fantasy.decisions.lineup import lineup_value
from fantasy.league_settings import LeagueSettings
from fantasy.league_state import LeagueSnapshot
from fantasy.orchestrator.models import Proposal, ProposalKind


def _accept_prob(opp_gain: float, opp_value_swing: float) -> float:
    """Opponent acceptance from BOTH their lineup gain and raw ROS value swing.

    ``opp_value_swing`` = ROS value they net (receive - give); strongly negative
    means they'd be fleeced, so they won't accept even if their lineup nominally
    improves. Both terms are in ROS-point units.
    """
    opp_net = opp_gain + opp_value_swing
    return 1.0 / (1.0 + math.exp(-opp_net / 8.0))


def recommend_trades(
    snap: LeagueSnapshot, board: pd.DataFrame, league: LeagueSettings,
    remaining_weeks: int, my_depth: int = 8, opp_depth: int = 10, top_k: int = 5,
    min_my_gain: float = 1.0,
) -> list[Proposal]:
    b = board.set_index("player_id")
    # Raw ROS points drive LINEUP value (points win weeks); ROS VOR drives cross-
    # positional VALUE/fairness (a high-scoring QB isn't "worth" more than a WR
    # once replacement level is accounted for).
    ros = {pid: float(b.loc[pid, "proj"]) * remaining_weeks for pid in b.index}
    ros_vor = {pid: float(b.loc[pid, "vor"]) * remaining_weeks for pid in b.index}
    pos = {pid: b.loc[pid, "position"] for pid in b.index}
    name = {pid: b.loc[pid, "player_display_name"] for pid in b.index}

    my = [p for p in snap.my_roster() if p in b.index]
    if not my:
        return []
    my_base = lineup_value(ros, pos, my, league)
    # Consider trading from my most VALUABLE assets (VOR), not highest raw scorers.
    my_cand = sorted(my, key=lambda p: ros_vor[p], reverse=True)[:my_depth]

    candidates = []
    for opp in snap.opponents():
        opp_roster = [p for p in snap.roster(opp) if p in b.index]
        if not opp_roster:
            continue
        opp_base = lineup_value(ros, pos, opp_roster, league)
        opp_cand = sorted(opp_roster, key=lambda p: ros_vor[p], reverse=True)[:opp_depth]
        for a in my_cand:  # I give a
            for c in opp_cand:  # I get c
                if pos[a] == pos[c] and abs(ros_vor[a] - ros_vor[c]) < 1:
                    continue
                my_after = lineup_value(ros, pos, [p for p in my if p != a] + [c], league)
                opp_after = lineup_value(ros, pos, [p for p in opp_roster if p != c] + [a], league)
                my_gain, opp_gain = my_after - my_base, opp_after - opp_base
                if my_gain < min_my_gain or opp_gain <= 0:
                    continue
                # Opponent receives `a`, gives `c`: their ROS VALUE (VOR) swing.
                p_acc = _accept_prob(opp_gain, ros_vor[a] - ros_vor[c])
                if p_acc < 0.2:  # they'd be fleeced on value — won't accept
                    continue
                candidates.append((my_gain * p_acc, my_gain, opp_gain, p_acc, opp, a, c))

    candidates.sort(reverse=True)
    seen, props = set(), []
    for score, my_gain, opp_gain, p_acc, opp, a, c in candidates:
        if (a, c) in seen:
            continue
        seen.add((a, c))
        props.append(
            Proposal(
                kind=ProposalKind.trade, season=snap.season, week=snap.week,
                team_id=snap.my_team_id,
                title=f"Trade {name[a]} → {name[c]} (w/ {snap.team_names.get(opp, opp)})",
                detail=(f"Send {name[a]} ({pos[a]}), get {name[c]} ({pos[c]}) "
                        f"from {snap.team_names.get(opp, opp)}.\n"
                        f"Your ROS lineup gain: +{my_gain:.1f} pts. "
                        f"Their gain: +{opp_gain:.1f} pts → accept prob ~{p_acc*100:.0f}%."),
                value=round(my_gain, 2), confidence=round(p_acc, 2),
                payload={"key_fields": {"give": a, "get": c, "with": opp},
                         "give": a, "get": c, "with_team": opp,
                         "my_gain": round(my_gain, 2), "accept_prob": round(p_acc, 2)},
            )
        )
        if len(props) >= top_k:
            break
    return props
