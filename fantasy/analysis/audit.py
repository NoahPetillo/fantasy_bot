"""Decision audit — replay a finished season and grade the manager's choices.

Reconstructs every executed transaction by diffing each week's ESPN box-score
rosters (add = appears, drop = disappears, trade = reciprocal cross-team move),
scores each on REALIZED league-scored points, and overlays the point-in-time
model (best available FA each week + win-win trades the model would have
proposed). Also grades start/sit: optimal hindsight lineup vs. what was actually
started — the points left on the bench.

``season_report`` returns a JSON-serializable dict consumed by both the CLI
(scripts/decision_audit.py) and the dashboard ("Season Report Card" panel).

Honest scope: EXECUTED moves only — declined/vetoed trade OFFERS leave no trace
in any ESPN read endpoint. "Realized" values an asset's production (with a
started-only column for the start/sit-aware view), not lineup execution.
"""

from __future__ import annotations

import logging
import statistics
from collections import defaultdict
from dataclasses import dataclass, field

import pandas as pd

from fantasy.data.ids import crosswalk
from fantasy.data.nfl import load_weekly
from fantasy.decisions.lineup import greedy_lineup
from fantasy.espn.client import EspnClient
from fantasy.league_settings import LeagueSettings
from fantasy.league_state import LeagueSnapshot
from fantasy.valuation.scoring import ScoringEngine

log = logging.getLogger(__name__)

STREAM_POS = {"K", "D/ST"}


@dataclass
class RP:  # roster player
    espn_id: str
    name: str
    position: str
    gsis: str | None


@dataclass
class Transition:
    week_to: int
    added: list[RP] = field(default_factory=list)
    dropped: list[RP] = field(default_factory=list)


@dataclass
class Trade:
    week_to: int
    other_team: int
    received: list[RP] = field(default_factory=list)
    sent: list[RP] = field(default_factory=list)


# ── roster reconstruction from weekly box scores ──────────────────────────────
def reconstruct(client: EspnClient, last_week: int):
    """rosters[w][team_id] -> {espn_id: RP};  box[espn_id][w] -> actual league pts;
    started[espn_id][w] -> pts only on weeks the player was in a STARTING slot."""
    xw = crosswalk()
    bench = {"BE", "IR"}
    rosters: dict[int, dict[int, dict[str, RP]]] = {}
    box: dict[str, dict[int, float]] = defaultdict(dict)
    started: dict[str, dict[int, float]] = defaultdict(dict)
    lg = client.league()
    for w in range(1, last_week + 1):
        try:
            games = lg.box_scores(w)
        except Exception as e:  # noqa: BLE001
            log.warning("box_scores(%d) failed: %s", w, e)
            continue
        rosters[w] = {}
        for m in games:
            for side, team in (("home_lineup", getattr(m, "home_team", None)),
                               ("away_lineup", getattr(m, "away_team", None))):
                tid = getattr(team, "team_id", None)
                if tid is None:
                    continue
                slot = rosters[w].setdefault(tid, {})
                for bp in getattr(m, side, []) or []:
                    eid = str(getattr(bp, "playerId", "") or "")
                    if not eid:
                        continue
                    slot[eid] = RP(eid, getattr(bp, "name", "?"),
                                   getattr(bp, "position", "?"), xw.from_espn(eid))
                    pts = getattr(bp, "points", None)
                    if pts is not None:
                        box[eid][w] = float(pts)
                        if str(getattr(bp, "slot_position", "")) not in bench:
                            started[eid][w] = float(pts)
    return rosters, box, started


def owners(rosters, week) -> dict[str, int]:
    out = {}
    for tid, players in rosters.get(week, {}).items():
        for eid in players:
            out[eid] = tid
    return out


def diff_team(rosters, team_id):
    """Per-team transitions + trades. A trade requires RECIPROCAL movement between
    the same two teams in one transition; one-directional cross-team movement is a
    waiver claim of a dropped player (add) or a drop someone else claimed."""
    transitions, trades = [], []
    weeks = sorted(rosters)
    for i in range(len(weeks) - 1):
        w, nxt = weeks[i], weeks[i + 1]
        prev, cur = set(rosters[w].get(team_id, {})), set(rosters[nxt].get(team_id, {}))
        gained, lost = cur - prev, prev - cur
        if not gained and not lost:
            continue
        op, oc = owners(rosters, w), owners(rosters, nxt)
        by_other: dict[int, Trade] = {}
        adds, drops = [], []
        for eid in gained:
            src = op.get(eid)
            (by_other.setdefault(src, Trade(nxt, src)).received if src not in (None, team_id)
             else adds).append(rosters[nxt][team_id][eid])
        for eid in lost:
            dst = oc.get(eid)
            (by_other.setdefault(dst, Trade(nxt, dst)).sent if dst not in (None, team_id)
             else drops).append(rosters[w][team_id][eid])
        for tr in by_other.values():
            if tr.received and tr.sent:               # reciprocal => real trade
                trades.append(tr)
            else:                                     # one-way => waiver/drop
                adds.extend(tr.received)
                drops.extend(tr.sent)
        if adds or drops:
            transitions.append(Transition(nxt, adds, drops))
    return transitions, trades


# ── realized points (ESPN box first = exact league scoring; nflverse fallback) ─
class Realized:
    def __init__(self, season: int, league: LeagueSettings, box, started, last_week: int):
        self.box, self.started, self.last_week = box, started, last_week
        eng = ScoringEngine(league)
        wk = load_weekly([season])
        wk = wk[wk["season"] == season].copy()
        wk["lpts"] = eng.score_dataframe(wk)
        self.nv = {(str(r.player_id), int(r.week)): float(r.lpts)
                   for r in wk.itertuples(index=False)}

    def wk(self, rp: RP, week: int) -> float:
        if week in self.box.get(rp.espn_id, {}):
            return self.box[rp.espn_id][week]                 # exact league scoring
        if rp.gsis and (rp.gsis, week) in self.nv:
            return self.nv[(rp.gsis, week)]                   # FA-week fallback
        return 0.0

    def span(self, rp: RP, frm: int, to: int) -> float:
        return sum(self.wk(rp, w) for w in range(frm, max(frm, to) + 1))

    def started_span(self, rp: RP, frm: int, to: int) -> float:
        s = self.started.get(rp.espn_id, {})
        return sum(s.get(w, 0.0) for w in range(frm, max(frm, to) + 1))

    def gsis_span(self, gsis: str, frm: int, to: int) -> float:
        return sum(self.nv.get((gsis, w), 0.0) for w in range(frm, max(frm, to) + 1))


# ── (b) start/sit audit — points left on the bench ────────────────────────────
def startsit_audit(rosters, box, started, my_id: int, league: LeagueSettings,
                   weeks: list[int]) -> dict:
    """For each week: optimal hindsight lineup (from actual points) vs what you
    actually started. The gap is points left on your bench."""
    out_weeks = []
    total_left = 0.0
    for w in weeks:
        roster = rosters.get(w, {}).get(my_id, {})
        if not roster:
            continue
        actual_started = {eid for eid in roster if w in started.get(eid, {})}
        started_pts = round(sum(started.get(eid, {}).get(w, 0.0) for eid in roster), 1)
        pts_by = {eid: box.get(eid, {}).get(w, 0.0) for eid in roster}
        pos_by = {eid: rp.position for eid, rp in roster.items()}
        optimal, _ = greedy_lineup(pts_by, pos_by, list(roster), league)
        left = round(optimal - started_pts, 1)
        if left > 0:
            total_left += left

        # Biggest single miss: highest-scoring benched player who'd have beaten an
        # eligible actual starter.
        biggest = None
        benched = sorted((eid for eid in roster if eid not in actual_started),
                         key=lambda e: pts_by[e], reverse=True)
        for be in benched:
            weaker = [s for s in actual_started
                      if pts_by[s] < pts_by[be] and _same_flex(pos_by[s], pos_by[be], league)]
            if weaker:
                worst = min(weaker, key=lambda s: pts_by[s])
                gain = round(pts_by[be] - pts_by[worst], 1)
                if gain > 0:
                    biggest = {"bench": roster[be].name, "bench_pts": round(pts_by[be], 1),
                               "over": roster[worst].name, "over_pts": round(pts_by[worst], 1),
                               "gain": gain}
                break
        out_weeks.append({"week": w, "started": started_pts, "optimal": round(optimal, 1),
                          "left": left, "biggest": biggest})
    n = len(out_weeks) or 1
    return {"weeks": out_weeks, "total_left_on_bench": round(total_left, 1),
            "avg_per_week": round(total_left / n, 1)}


def model_startsit(service, client, season, rosters, box, started, my_id, league,
                   reg_end, realized: "Realized", cache) -> dict:
    """Decision-relevant start/sit: for each week build the PROJECTION-optimal lineup
    (point-in-time board) and sum the players' ACTUAL realized points, vs what you
    actually started. Positive = following the model would have scored more. Unlike
    hindsight 'left on bench', this is achievable pre-game."""
    total_gain, weeks = 0.0, []
    for w in range(1, reg_end + 1):
        roster = rosters.get(w, {}).get(my_id, {})
        if not roster:
            continue
        board = board_for(service, client, season, w, cache)
        b = board.set_index("player_id")
        proj_by, pos_by = {}, {}
        for eid, rp in roster.items():
            proj_by[eid] = float(b.loc[rp.gsis, "proj"]) if (rp.gsis and rp.gsis in b.index) else 0.0
            pos_by[eid] = rp.position
        _, chosen = greedy_lineup(proj_by, pos_by, list(roster), league)
        model_real = sum(realized.wk(rp, w) for eid, rp in roster.items() if eid in chosen)
        actual = sum(started.get(eid, {}).get(w, 0.0) for eid in roster)
        gain = round(model_real - actual, 1)
        total_gain += gain
        weeks.append({"week": w, "gain": gain})
    n = len(weeks) or 1
    return {"total_gain": round(total_gain, 1), "avg_per_week": round(total_gain / n, 1),
            "weeks": weeks}


def _same_flex(pos_a: str, pos_b: str, league: LeagueSettings) -> bool:
    if pos_a == pos_b:
        return True
    from fantasy.espn.stat_ids import FLEX_ELIGIBILITY
    for slot in league.roster.starter_slots:
        elig = FLEX_ELIGIBILITY.get(slot)
        if elig and pos_a in elig and pos_b in elig:
            return True
    return False


# ── model overlay helpers ─────────────────────────────────────────────────────
def board_for(service, client, season, week, cache):
    if week not in cache:
        try:
            ep = client.week_projections(week)
        except Exception:  # noqa: BLE001
            ep = {}
        cache[week] = service.project(season, week, espn_proj=ep or None)
    return cache[week]


def snapshot_from_rosters(rosters, season, week, my_id, board, league) -> LeagueSnapshot:
    b = board.set_index("player_id")
    teams, names, positions = {}, {}, {}
    for tid, players in rosters.get(week, {}).items():
        ids = []
        for rp in players.values():
            pid = rp.gsis or f"espn:{rp.espn_id}"
            ids.append(pid)
            names[pid] = rp.name
            positions[pid] = rp.position
        teams[tid] = ids
    rostered = {p for ids in teams.values() for p in ids}
    fas = [pid for pid in b.index if pid not in rostered]
    for pid in fas:
        names.setdefault(pid, b.loc[pid, "player_display_name"])
        positions.setdefault(pid, b.loc[pid, "position"])
    return LeagueSnapshot(season, week, my_id, teams, fas, names, positions,
                          {t: league.faab_budget for t in teams},
                          {t: f"Team {t}" for t in teams})


def model_trade_scan(service, client, season, rosters, my_id, league, reg_end,
                     cache) -> dict:
    """Best win-win trade the model would have proposed each week, with its realized
    regular-season outcome. Reports the distribution (not a cherry-picked week)."""
    from fantasy.decisions.trades import recommend_trades

    realized = cache["_realized"]
    weeks_out, outcomes, best = [], [], None
    for w in range(2, reg_end):
        if w not in rosters:
            continue
        board = board_for(service, client, season, w, cache)
        snap = snapshot_from_rosters(rosters, season, w, my_id, board, league)
        props = recommend_trades(snap, board, league, max(reg_end - w + 1, 1), top_k=1)
        if not props:
            continue
        p = props[0]
        give, get = p.payload["give"], p.payload["get"]
        net = round(realized.gsis_span(get, w, reg_end) - realized.gsis_span(give, w, reg_end), 1)
        b = board.set_index("player_id")
        gn = b.loc[get, "player_display_name"] if get in b.index else get
        vn = b.loc[give, "player_display_name"] if give in b.index else give
        row = {"week": w, "give": vn, "get": gn,
               "accept": round(float(p.payload.get("accept_prob", p.confidence)), 2),
               "realized": net}
        weeks_out.append(row)
        outcomes.append(net)
        if best is None or net > best["realized"]:
            best = row
    summary = {"weeks": weeks_out, "best_trade": best}
    if outcomes:
        summary.update({
            "n": len(outcomes),
            "win_rate": round(sum(1 for x in outcomes if x > 0) / len(outcomes), 2),
            "mean": round(statistics.mean(outcomes), 1),
            "best": round(max(outcomes), 1), "worst": round(min(outcomes), 1),
        })
    return summary


# ── the report card ───────────────────────────────────────────────────────────
def season_report(client: EspnClient, league: LeagueSettings, my_id: int,
                  season: int, service=None, through: int | None = None) -> dict:
    """Full retrospective report card for ``my_id`` over ``season``. Heavy: pulls
    every week's box scores; if ``service`` is given, also runs the model overlay."""
    lg = client.league()
    last_week = min(through or 17, getattr(lg, "current_week", 17))
    reg_end = league.regular_season_weeks or 14
    rosters, box, started = reconstruct(client, last_week)
    realized = Realized(season, league, box, started, last_week)
    transitions, trades = diff_team(rosters, my_id)

    # Waivers — per-player skill/stream tagging (a bundled kicker must NOT inflate
    # the skill number); both asset points and started-only points.
    nets = {"skill_asset": 0.0, "skill_started": 0.0,
            "stream_asset": 0.0, "stream_started": 0.0}
    moves = []
    for tr in sorted(transitions, key=lambda x: x.week_to):
        m = {"week": tr.week_to, "added": [], "dropped": [], "net": 0.0}
        for p in tr.added:
            a = realized.span(p, tr.week_to, reg_end)
            s = realized.started_span(p, tr.week_to, reg_end)
            bk = "stream" if p.position in STREAM_POS else "skill"
            nets[f"{bk}_asset"] += a
            nets[f"{bk}_started"] += s
            m["net"] += a
            m["added"].append({"name": p.name, "pos": p.position,
                               "asset": round(a, 1), "started": round(s, 1)})
        for p in tr.dropped:
            a = realized.span(p, tr.week_to, reg_end)
            s = realized.started_span(p, tr.week_to, reg_end)
            bk = "stream" if p.position in STREAM_POS else "skill"
            nets[f"{bk}_asset"] -= a
            nets[f"{bk}_started"] -= s
            m["net"] -= a
            m["dropped"].append({"name": p.name, "pos": p.position, "asset": round(a, 1)})
        m["net"] = round(m["net"], 1)
        moves.append(m)
    waiver = {k: round(v, 1) for k, v in nets.items()}
    waiver["moves"] = moves

    startsit = startsit_audit(rosters, box, started, my_id, league, list(range(1, reg_end + 1)))

    model_trades = {}
    if service is not None:
        cache = {"_realized": realized}
        try:
            model_trades = model_trade_scan(service, client, season, rosters, my_id,
                                            league, reg_end, cache)
        except Exception as e:  # noqa: BLE001
            log.warning("model trade scan failed: %s", e)
        try:
            startsit["model"] = model_startsit(service, client, season, rosters, box,
                                               started, my_id, league, reg_end, realized, cache)
        except Exception as e:  # noqa: BLE001
            log.warning("model start/sit failed: %s", e)

    all_teams = {t for by in rosters.values() for t in by}
    league_trades = sum(len(diff_team(rosters, t)[1]) for t in all_teams) // 2
    return {
        "season": season, "team_id": my_id, "reg_end": reg_end, "through": last_week,
        "trades_made": len(trades),
        "trades": [{"week": t.week_to, "received": [p.name for p in t.received],
                    "sent": [p.name for p in t.sent]} for t in trades],
        "league_trades": league_trades,
        "waiver": waiver,
        "startsit": startsit,
        "model_trades": model_trades,
    }
