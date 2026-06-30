"""Assemble the dashboard payload from the live decision layer.

Heavy compute (model + ESPN) runs once in scripts/dashboard.py, which writes a
snapshot to data/dashboard.json; the API serves that snapshot and overlays live
proposal statuses from the store. Keeps the page instant while staying real.
"""

from __future__ import annotations

import json
import logging

from fantasy.config import settings
from fantasy.data.ids import norm_name
from fantasy.decisions.lineup import optimize_lineup
from fantasy.decisions.startsit import recommend_lineup
from fantasy.decisions.trades import recommend_trades
from fantasy.decisions.waivers import recommend_waivers
from fantasy.league_state import build_dryrun_snapshot, build_live_snapshot
from fantasy.orchestrator.store import Store

log = logging.getLogger(__name__)

SNAPSHOT = settings.data_dir / "dashboard.json"


def assemble(service, league, store: Store, season: int, week: int, client=None,
             with_report: bool = True) -> dict:
    from fantasy.news.experts.adjust import priority_boosts
    from fantasy.orchestrator.cycle import fetch_expert_signals

    espn_proj = client.week_projections(week) if client else None
    fused = fetch_expert_signals()  # corroboration-gated; [] offseason/offline
    board = service.project(season, week, espn_proj=espn_proj, fused_signals=fused)
    snap = (build_live_snapshot(client, league, season, week) if client
            else build_dryrun_snapshot(board, league, season, week))
    rem = service.remaining_weeks(week)
    b = board.set_index("player_id")

    lineup_props = recommend_lineup(snap, board, league)
    waiver_props = recommend_waivers(snap, board, league, rem, boosts=priority_boosts(fused))
    trade_props = recommend_trades(snap, board, league, rem)
    if settings.prioritize_trades:
        for p in trade_props:
            p.payload["priority"] = True
    for p in lineup_props + waiver_props + trade_props:
        store.add(p)

    def nm(pid):
        return b.loc[pid, "player_display_name"] if pid in b.index else pid

    def pos(pid):
        return b.loc[pid, "position"] if pid in b.index else "?"

    waivers = [{
        "id": p.id, "add": nm(p.payload["add"]), "drop": nm(p.payload["drop"]),
        "pos": pos(p.payload["add"]), "bid": p.payload.get("faab_bid") or None,
        "reason": p.detail.replace("\n", " "), "value": round(p.value, 1),
        "conf": p.confidence,
    } for p in waiver_props]

    trades = [{
        "id": p.id, "give": nm(p.payload["give"]), "get": nm(p.payload["get"]),
        "reason": p.detail.replace("\n", " "), "my_gain": round(p.value, 1),
        "accept": p.payload.get("accept_prob", p.confidence),
        "priority": bool(p.payload.get("priority")),
    } for p in trade_props]

    # Optimal lineup rows for the dashboard.
    mine = board[board["player_id"].isin(snap.my_roster())]
    rows, total = [], 0.0
    if not mine.empty:
        players = [(r.player_id, r.position, float(r.proj)) for r in mine.itertuples(index=False)]
        lu = optimize_lineup(players, league)
        mb = mine.set_index("player_id")
        for slot, pids in lu.items():
            for pid in pids:
                r = mb.loc[pid]
                total += float(r["proj"])
                rows.append({"slot": slot, "name": r["player_display_name"], "pos": r["position"],
                             "proj": round(float(r["proj"]), 1),
                             "floor": r.get("floor"), "ceiling": r.get("ceiling")})

    standings = _standings(client, snap.my_team_id) if client else []
    feed = _feed(store, fused)
    actions = [{"id": p.id, "kind": p.kind.value, "title": p.title,
                "value": round(p.value, 1), "status": p.status.value}
               for p in store.list(limit=40)]

    board_index = {norm_name(r.player_display_name): {
        "name": r.player_display_name, "pos": r.position,
        "vor": round(float(r.vor), 1), "proj": round(float(r.proj), 1)}
        for r in board.itertuples(index=False)}

    report = _report_card(service, league, snap, season, client) if with_report else None

    pending = len([a for a in actions if a["status"] == "proposed"])
    kpis = [
        {"label": "Week", "value": str(week), "sub": f"{league.scoring_format.value} · {league.team_count}-team"},
        {"label": "Proj lineup", "value": f"{total:.0f}", "sub": "optimal start/sit", "accent": True},
        {"label": "Waiver targets", "value": str(len(waivers)), "sub": "ranked upgrades"},
        {"label": "Pending", "value": str(pending), "sub": "awaiting your approval"},
    ]

    return {
        "team": {"name": snap.team_names.get(snap.my_team_id, "My Team"),
                 "league": league.summary(), "week": week, "mode": settings.execution_mode.value,
                 "prioritize_trades": settings.prioritize_trades},
        "kpis": kpis, "waivers": waivers, "trades": trades, "lineup": rows,
        "lineup_total": round(total, 1), "standings": standings, "feed": feed,
        "actions": actions, "board_index": board_index, "report": report,
    }


def _report_card(service, league, snap, season: int, client) -> dict | None:
    """Retrospective season report card (the decision audit) for the dashboard.
    Heavy (pulls every week's box scores) — best-effort; never breaks the page."""
    if client is None:
        return None
    try:
        from fantasy.analysis.audit import season_report
        return season_report(client, league, snap.my_team_id, season, service=service)
    except Exception as e:  # noqa: BLE001
        log.warning("report card unavailable: %s", e)
        return None


def _standings(client, my_team_id) -> list[dict]:
    try:
        teams = sorted(client.teams(), key=lambda t: (getattr(t, "wins", 0),
                       getattr(t, "points_for", 0)), reverse=True)
        return [{"rank": i + 1, "team": getattr(t, "team_name", "?"),
                 "w": getattr(t, "wins", 0), "l": getattr(t, "losses", 0),
                 "pf": round(getattr(t, "points_for", 0), 1),
                 "me": getattr(t, "team_id", None) == my_team_id}
                for i, t in enumerate(teams)]
    except Exception as e:  # noqa: BLE001
        log.warning("standings unavailable: %s", e)
        return []


def _feed(store: Store, fused: list) -> list[dict]:
    items = []
    for f in (fused or [])[:12]:
        items.append({"title": f"{f.player_name} — {f.event_type.value.replace('_', ' ')}",
                      "detail": f.rationale, "corroborated": f.corroborated})
    # Plus any alert proposals already logged.
    for p in store.list(kind="alert", limit=10):
        items.append({"title": p.title, "detail": p.detail, "corroborated": p.confidence >= 0.7})
    return items


def write_snapshot(payload: dict) -> None:
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT.write_text(json.dumps(payload))


def read_snapshot() -> dict | None:
    try:
        return json.loads(SNAPSHOT.read_text())
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None


def analyze_trade(give: str, get: str, board_index: dict) -> dict:
    g = board_index.get(norm_name(give))
    h = board_index.get(norm_name(get))
    if not g or not h:
        missing = give if not g else get
        return {"error": f"'{missing}' not found on the value board."}
    net = round(h["vor"] - g["vor"], 1)
    diff = abs(net)
    fairness = "even" if diff <= 3 else ("slightly lopsided" if diff <= 8 else "lopsided")
    return {"give_vor": g["vor"], "get_vor": h["vor"], "net": net, "fairness": fairness}
