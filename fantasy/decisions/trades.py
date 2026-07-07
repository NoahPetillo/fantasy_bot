"""Trade engine — auto-discovery of win-win offers + manual package analysis.

Two entry points, both valuing a trade by the rest-of-season *starting-lineup*
gain (greedy lineup value on ROS-scaled projections) rather than a raw point sum,
so roster fit is priced in — a player who can't crack your lineup adds little:

- ``recommend_trades``: scans every opponent for win-win 1-for-1 swaps and ranks
  them by ``my_gain * P(accept)`` (acceptance rises with the opponent's own lineup
  gain — offers that genuinely help them too). This is the engine behind the
  proactive "trades you should propose" notifications.
- ``evaluate_trade_package``: scores an arbitrary N-for-M package the user builds
  in the dashboard's Trade Analyzer, against their real roster + league rules.
  Reports lineup gain, the raw-points-vs-lineup contrast, VOR fairness, a depth
  (bench-insurance) term, roster legality, and opponent accept-likelihood.

Market-value fairness (FantasyCalc) and a learned per-manager acceptance model are
later refinements; ROS lineup gain is the honest v1 signal.
"""

from __future__ import annotations

import math

import pandas as pd

from fantasy.decisions.lineup import greedy_lineup, lineup_value
from fantasy.league_settings import LeagueSettings
from fantasy.league_state import LeagueSnapshot
from fantasy.orchestrator.models import Proposal, ProposalKind

# Bench players only score when a starter is hurt/on bye, so surplus depth is
# worth a fraction of a starter's ROS value. Used to price the depth a package
# trade gains or gives up — kept small and reported separately from lineup gain.
DEPTH_WEIGHT = 0.15


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
    # once replacement level is accounted for). Both are bye-aware.
    from fantasy.decisions.ros import ros_maps
    ros, ros_vor = ros_maps(board, league, snap.season, snap.week, remaining_weeks)
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


def _depth_value(roster: list[str], starters: set[str], ros_vor: dict[str, float]) -> float:
    """Insurance value of the surplus (non-starting) players on a roster: a small
    fraction of each bench body's ROS VOR. A below-replacement body (VOR < 0) is
    worth ~0 as depth, so we clip at 0."""
    return DEPTH_WEIGHT * sum(max(0.0, ros_vor.get(p, 0.0)) for p in roster if p not in starters)


def evaluate_trade_package(
    my_roster: list[str], counter_roster: list[str] | None,
    give: list[str], get: list[str],
    ros: dict[str, float], ros_vor: dict[str, float], pos: dict[str, str],
    league: LeagueSettings, bench_size: int, ir_size: int = 0,
    single_counterparty: bool = True, names: dict[str, str] | None = None,
) -> dict:
    """Evaluate an arbitrary N-for-M trade for MY team, roster-fit-aware.

    The headline is the change in my best legal STARTING lineup over the rest of
    the season (``lineup_delta``), computed with the same greedy engine as the
    auto-suggester. Because a lineup only rewards players who crack a starting
    slot, acquiring two players when you can only start one adds far less than the
    raw point sum — and a single "big hitter" who takes a slot can be worth more
    than two who ride the bench. Depth (bench insurance) and cross-positional
    value (VOR) are reported alongside so the trade-off is visible, not hidden.

    All maps are expected ROS-scaled (proj/vor × remaining weeks). ``ros``/``ros_vor``
    default to 0.0 for players without a projection (many K/DST/rookies).
    """
    names = names or {}
    give_set, get_set = set(give), set(get)
    before = [p for p in my_roster]
    after = [p for p in before if p not in give_set] + list(get)

    lineup_before, starters_before = greedy_lineup(ros, pos, before, league)
    lineup_after, starters_after = greedy_lineup(ros, pos, after, league)
    lineup_delta = round(lineup_after - lineup_before, 1)

    give_pts = sum(ros.get(a, 0.0) for a in give)
    get_pts = sum(ros.get(c, 0.0) for c in get)
    give_vor = sum(ros_vor.get(a, 0.0) for a in give)
    get_vor = sum(ros_vor.get(c, 0.0) for c in get)
    points_sum_delta = round(get_pts - give_pts, 1)
    vor_delta = round(get_vor - give_vor, 1)

    depth_delta = round(_depth_value(after, starters_after, ros_vor)
                        - _depth_value(before, starters_before, ros_vor), 1)

    diff = abs(vor_delta)
    fairness = "even" if diff <= 3 else ("slightly lopsided" if diff <= 8 else "lopsided")

    # Full roster capacity, incl. IR — snap rosters count IR players, so the cap must too,
    # else every trade in an IR league would falsely read as over the limit.
    cap = league.roster.total_starters + bench_size + ir_size
    need_to_drop = max(0, (len(my_roster) - len(give) + len(get)) - cap)

    accept_prob = None
    if single_counterparty and counter_roster:
        opp_before = lineup_value(ros, pos, counter_roster, league)
        opp_after = lineup_value(
            ros, pos, [p for p in counter_roster if p not in get_set] + list(give), league)
        # Opponent receives what I `give` and gives up what I `get`.
        accept_prob = round(_accept_prob(opp_after - opp_before, give_vor - get_vor), 2)

    def _line(pid: str, starter_set: set[str]) -> dict:
        return {"id": pid, "name": names.get(pid, pid), "pos": pos.get(pid, "?"),
                "ros_proj": round(ros.get(pid, 0.0), 1), "ros_vor": round(ros_vor.get(pid, 0.0), 1),
                "starter": pid in starter_set}

    give_detail = [_line(a, starters_before) for a in give]
    get_detail = [_line(c, starters_after) for c in get]

    notes = []
    unpriced = [names.get(p, p) for p in list(give) + list(get) if p not in ros]
    if unpriced:
        notes.append(f"No projection for {', '.join(unpriced)} — treated as replacement level.")
    benched_in = [d["name"] for d in get_detail if not d["starter"]]
    if benched_in:
        notes.append(f"{', '.join(benched_in)} wouldn't crack your starting lineup — "
                     "those points sit on your bench.")
    if need_to_drop:
        notes.append(f"This trade leaves you {need_to_drop} over the roster limit — "
                     f"you'd need to drop {need_to_drop} more player(s).")
    if not single_counterparty:
        notes.append("Players come from multiple teams — shown for analysis, not as one offer.")

    # Verdict keys off the depth-adjusted gain so it can't read "neutral" while the
    # numbers beside it show real bench insurance given up (or gained) for no lineup change.
    adjusted_delta = round(lineup_delta + depth_delta, 1)
    verdict = "favorable" if adjusted_delta > 2 else ("unfavorable" if adjusted_delta < -2 else "neutral")

    return {
        "lineup_delta": lineup_delta, "lineup_before": round(lineup_before, 1),
        "lineup_after": round(lineup_after, 1), "points_sum_delta": points_sum_delta,
        "vor_delta": vor_delta, "depth_delta": depth_delta, "adjusted_delta": adjusted_delta,
        "fairness": fairness, "accept_prob": accept_prob, "need_to_drop": need_to_drop,
        "verdict": verdict, "give": give_detail, "get": get_detail, "notes": notes,
    }
