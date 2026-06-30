"""Validation: self-play A/B and real-draft replay.

run_selfplay: at the same seat, with identical opponents/seed/board, compare our
agent vs an ADP-autopick agent — isolating draft skill. Reports mean realized-point
delta with a bootstrap CI.

replay_real_draft: at each of the user's ACTUAL pick slots, what would the engine
have recommended, and did that player outscore who was actually taken (by realized
that-season points)? The credibility artifact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fantasy.data.ids import crosswalk
from fantasy.data.nfl import season_totals
from fantasy.draft.board import build_board, build_replay_board
from fantasy.draft.recommend import recommend
from fantasy.draft.simulator import AdpAgent, OurAgent, run_draft, score_rosters
from fantasy.draft.state import DraftState
from fantasy.league_settings import LeagueSettings
from fantasy.valuation.scoring import ScoringEngine


def _rounds(league: LeagueSettings) -> int:
    return league.roster.total_starters + league.roster.bench_size


def run_selfplay(league: LeagueSettings, seasons: list[int], n_per_season: int = 24,
                 strategy: str = "none", teams: int = 12) -> dict:
    rounds = _rounds(league)
    pick_order = list(range(1, teams + 1))
    deltas = []
    for season in seasons:
        board = build_board(season, league, teams=teams)
        for i in range(n_per_season):
            seat = (i % teams) + 1
            base = {t: AdpAgent() for t in pick_order}
            ours = dict(base)
            ours[seat] = OurAgent(strategy)
            a = score_rosters(run_draft(league, board, ours, pick_order, rounds, seed=i), season, league)
            b = score_rosters(run_draft(league, board, base, pick_order, rounds, seed=i), season, league)
            deltas.append(a[seat] - b[seat])
    arr = np.array(deltas, float)
    boot = [np.mean(np.random.default_rng(s).choice(arr, len(arr))) for s in range(1000)]
    return {
        "n": len(arr), "mean_delta": float(arr.mean()),
        "ci95": (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))),
        "win_rate": float((arr > 0).mean()),
    }


def replay_real_draft(client, season: int, league: LeagueSettings, strategy: str = "none",
                      teams: int = 12) -> pd.DataFrame:
    xw = crosswalk()
    tot = season_totals([season], ScoringEngine(league))
    actual_pts = dict(zip(tot["player_id"], tot["pts"]))

    from fantasy.config import settings

    raw = client._raw(["mSettings"])
    pick_order = raw["settings"]["draftSettings"].get("pickOrder") or list(range(1, teams + 1))
    my_team_id = settings.espn_team_id or pick_order[0]
    rounds = _rounds(league)

    # Real picks in overall order, mapped to gsis.
    seq = []
    for p in client.draft():
        overall = (p.round_num - 1) * teams + p.round_pick
        gid = xw.from_espn(getattr(p, "playerId", None))
        seq.append((overall, getattr(getattr(p, "team", None), "team_id", None),
                    gid, getattr(p, "playerName", "?")))
    seq.sort()

    # Build the replay board from the real draft (its order = ADP).
    picks_df = pd.DataFrame(
        [{"player_id": g, "player_display_name": nm,
          "position": xw.gsis_to_pos.get(g), "adp": o}
         for (o, t, g, nm) in seq if g],
    ).dropna(subset=["position"])
    board = build_replay_board(picks_df, season, league)
    board_pos = board.set_index("player_id")["position"].to_dict()

    # Engine drafts a COHERENT roster: at each of my slots it picks best-available
    # (removing its own prior picks); opponents are held to their real picks.
    engine_picks: list[str] = []
    rows = []
    for overall, team, gid, name in seq:
        if team != my_team_id:
            continue
        opp_taken = [(o, t, g) for (o, t, g, _) in seq if o < overall and t != my_team_id and g]
        picks_so_far = opp_taken + [(0, my_team_id, ep) for ep in engine_picks]
        state = DraftState(league, list(pick_order), rounds, board, my_team_id=my_team_id,
                           picks=picks_so_far)
        recs = recommend(state, strategy, top=3)
        if not recs:
            continue
        eng = recs[0]
        engine_picks.append(eng.player_id)
        rows.append({
            "round": (overall - 1) // teams + 1, "overall": overall,
            "actual": name, "actual_pts": round(actual_pts.get(gid, 0.0), 1),
            "engine": eng.name, "engine_pts": round(actual_pts.get(eng.player_id, 0.0), 1),
            "delta": round(actual_pts.get(eng.player_id, 0.0) - actual_pts.get(gid, 0.0), 1),
            "why": eng.reason,
        })
    table = pd.DataFrame(rows)

    # Roster-level comparison: best season-long lineup of each drafted roster.
    from fantasy.decisions.lineup import greedy_lineup
    my_actual = [g for (o, t, g, _) in seq if t == my_team_id and g]
    eng_lineup = greedy_lineup({p: actual_pts.get(p, 0.0) for p in engine_picks},
                               board_pos, engine_picks, league)[0]
    act_lineup = greedy_lineup({p: actual_pts.get(p, 0.0) for p in my_actual},
                               {**board_pos, **{g: xw.gsis_to_pos.get(g) for g in my_actual}},
                               my_actual, league)[0]
    table.attrs["engine_roster_pts"] = eng_lineup
    table.attrs["actual_roster_pts"] = act_lineup
    return table
