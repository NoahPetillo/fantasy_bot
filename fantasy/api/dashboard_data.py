"""Assemble the dashboard payload from the live decision layer.

Heavy compute (model + ESPN) runs once in scripts/dashboard.py, which writes a
snapshot to data/dashboard.json; the API serves that snapshot and overlays live
proposal statuses from the store. Keeps the page instant while staying real.
"""

from __future__ import annotations

import json
import logging

from typing import Protocol

from fantasy.config import settings
from fantasy.data.ids import norm_name
from fantasy.decisions.lineup import optimize_lineup
from fantasy.decisions.startsit import recommend_lineup
from fantasy.decisions.trades import recommend_trades
from fantasy.decisions.waivers import recommend_waivers
from fantasy.league_state import build_dryrun_snapshot, build_live_snapshot

log = logging.getLogger(__name__)


class ProposalStore(Protocol):
    """The proposal-log surface ``assemble`` needs — satisfied by both the legacy
    SQLite ``Store`` (single-tenant) and the per-user ``PgProposalStore``."""

    def add(self, p) -> bool: ...
    def by_key(self, idempotency_key: str): ...
    def get(self, proposal_id: str): ...
    def merge_payload(self, proposal_id: str, updates: dict) -> None: ...
    def set_status(self, proposal_id: str, status, notify_ref: str | None = None) -> None: ...
    def list(self, status=None, season=None, week=None, kind=None, limit: int = 200): ...


def snapshot_path(league_id: int | str | None = None) -> "Path":
    from pathlib import Path
    if league_id is None:
        return settings.data_dir / "dashboard.json"
    return settings.data_dir / f"dashboard_{league_id}.json"


def assemble(service, league, store: ProposalStore, season: int, week: int, client=None,
             with_report: bool = True, my_team_id: int | None = None) -> dict:
    from fantasy.news.experts.adjust import priority_boosts
    from fantasy.orchestrator.cycle import fetch_expert_signals

    espn_proj = client.week_projections(week) if client else None
    fused = fetch_expert_signals()  # corroboration-gated; [] offseason/offline
    board = service.project(season, week, espn_proj=espn_proj, fused_signals=fused)
    if board.empty and client is not None:
        # No player data for this (season, week) yet — e.g. the current season
        # before its weekly stats are published (preseason). There's nothing to
        # project a lineup/waiver/trade from, so serve the instant shell view
        # rather than fabricating recommendations from an empty board.
        log.info("Empty value board for %s wk%s — returning shell view.", season, week)
        return shell_snapshot(client, league, season, week, my_team_id)
    snap = (build_live_snapshot(client, league, season, week, my_team_id=my_team_id) if client
            else build_dryrun_snapshot(board, league, season, week))
    rem = service.remaining_weeks(week)
    b = board.set_index("player_id")

    lineup_props = recommend_lineup(snap, board, league)
    waiver_props = recommend_waivers(snap, board, league, rem, boosts=priority_boosts(fused))
    trade_props = recommend_trades(snap, board, league, rem)
    if settings.prioritize_trades:
        for p in trade_props:
            p.payload["priority"] = True
    lid = getattr(league, "league_id", None)

    def _name(pid):
        return str(b.loc[pid, "player_display_name"]) if pid in b.index else pid

    def persist(p):
        """Persist + return the CANONICAL proposal. On a rebuild the same advice has
        the same idempotency key, so add() is a no-op and we must reference the row
        already in the store (with its live status) — otherwise the dashboard would
        link to a fresh id that approve/reject can't find."""
        if lid is not None:
            p.payload["league_id"] = lid  # so a later "verify on ESPN" knows the league
        ids = [p.payload[k] for k in ("add", "drop", "give", "get")
               if isinstance(p.payload.get(k), str)]
        ids += p.payload.get("key_fields", {}).get("starters", []) or []
        if ids:  # names so "verify on ESPN" reads cleanly even for off-roster players
            p.payload["names"] = {i: _name(i) for i in ids}
        if store.add(p):
            return p
        existing = store.by_key(p.idempotency_key)
        if existing is None:
            return p
        store.merge_payload(existing.id, p.payload)  # keep league_id/priority current
        return store.get(existing.id) or existing

    lineup_props = [persist(p) for p in lineup_props]
    waiver_props = [persist(p) for p in waiver_props]
    trade_props = [persist(p) for p in trade_props]

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

    trade_block = _trade_block(snap, b, league, rem)

    report = _report_card(service, league, snap, season, client) if with_report else None
    from fantasy.orchestrator.influence import influence_stats
    influence = influence_stats(store, season=season, team_id=snap.my_team_id)

    pending = len([a for a in actions if a["status"] == "proposed"])
    kpis = [
        {"label": "Week", "value": str(week), "sub": f"{league.scoring_format.value} · {league.team_count}-team"},
        {"label": "Proj lineup", "value": f"{total:.0f}", "sub": "optimal start/sit", "accent": True},
        {"label": "Waiver targets", "value": str(len(waivers)), "sub": "ranked upgrades"},
        {"label": "Pending", "value": str(pending), "sub": "awaiting your approval"},
    ]

    return {
        "team": {"name": snap.team_names.get(snap.my_team_id, "My Team"),
                 "league": league.summary(), "week": week, "season": season,
                 "mode": settings.execution_mode.value,
                 "prioritize_trades": settings.prioritize_trades,
                 "league_id": getattr(league, "league_id", None), "team_id": snap.my_team_id},
        "kpis": kpis, "waivers": waivers, "trades": trades, "lineup": rows,
        "lineup_total": round(total, 1), "standings": standings, "feed": feed,
        "actions": actions, "board_index": board_index, "trade": trade_block,
        "report": report, "influence": influence,
        "league_settings": {                      # compact, so the chatbot can answer offline
            "summary": league.summary(),
            "scoring": {k: v for k, v in league.scoring.items() if v},
            "te_premium": dict(getattr(league, "position_reception_bonus", {}) or {}),
            "roster": league.roster.starter_slots,
        },
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


def _feed(store: ProposalStore, fused: list) -> list[dict]:
    items = []
    for f in (fused or [])[:12]:
        items.append({"title": f"{f.player_name} — {f.event_type.value.replace('_', ' ')}",
                      "detail": f.rationale, "corroborated": f.corroborated})
    # Plus any alert proposals already logged.
    for p in store.list(kind="alert", limit=10):
        items.append({"title": p.title, "detail": p.detail, "corroborated": p.confidence >= 0.7})
    return items


def write_snapshot(payload: dict, league_id: int | str | None = None) -> None:
    p = snapshot_path(league_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload))


def shell_snapshot(client, league, season: int, week: int, my_team_id: int | None) -> dict:
    """Cheap, instant payload for a freshly-added league: settings, standings, and
    your roster — no model, no projections. Lets a new (even pre-draft) league show
    up immediately; the full analysis is built on demand in the background."""
    standings = _standings(client, my_team_id)
    me = next((s["team"] for s in standings if s.get("me")), "My Team")
    drafted = bool(standings) and any(s.get("w", 0) or s.get("l", 0) for s in standings)
    note = ("Full analysis not built yet — tap “Build analysis”."
            if drafted else "Season hasn’t started (no draft yet). Settings are loaded; "
            "build the full analysis once rosters exist.")
    return {
        "team": {"name": me, "league": league.summary(), "week": week, "season": season,
                 "mode": settings.execution_mode.value, "team_id": my_team_id,
                 "prioritize_trades": settings.prioritize_trades,
                 "shell": True, "status": note, "league_id": league.league_id},
        "kpis": [
            {"label": "League", "value": f"{league.team_count}-team", "sub": league.scoring_format.value},
            {"label": "Status", "value": "Preseason" if not drafted else "Ready", "sub": "shell view", "accent": True},
            {"label": "Teams", "value": str(len(standings)), "sub": "in league"},
            {"label": "Analysis", "value": "—", "sub": "build on demand"},
        ],
        "waivers": [], "trades": [], "lineup": [], "lineup_total": 0,
        "standings": standings, "feed": [], "actions": [], "board_index": {},
        "report": None, "influence": None,
    }


def _trade_block(snap, b, league, remaining_weeks: int) -> dict:
    """Everything the manual Trade Analyzer needs, baked into the snapshot so the
    analyze endpoint stays instant (no live model/ESPN call). Players are keyed by
    id and carry their owning team, so the picker can scope "give" to my roster and
    "get" to other teams, and the analyzer can price a package against real rosters.
    Includes every ROSTERED player (union of all teams) — free agents are a waiver
    concern, not tradeable. proj/vor default to 0 for ids without a projection."""
    players: dict[str, dict] = {}
    for tid, pids in snap.teams.items():
        for pid in pids:
            row = b.loc[pid] if pid in b.index else None
            players[pid] = {
                "name": str(row["player_display_name"]) if row is not None else snap.names.get(pid, pid),
                "pos": str(row["position"]) if row is not None else snap.positions.get(pid, "?"),
                "proj": round(float(row["proj"]), 1) if row is not None else 0.0,
                "vor": round(float(row["vor"]), 1) if row is not None else 0.0,
                "team_id": int(tid),
            }
    return {
        "my_team_id": int(snap.my_team_id),
        "remaining_weeks": int(remaining_weeks),
        "team_count": league.team_count,
        "roster_slots": league.roster.starter_slots,
        "bench_size": league.roster.bench_size,
        "ir_size": int(league.roster.slots.get("IR", 0)),
        "team_names": {str(t): n for t, n in snap.team_names.items()},
        "teams": {str(t): list(pids) for t, pids in snap.teams.items()},
        "players": players,
    }


def analyze_trade(give, get, payload: dict) -> dict:
    """Evaluate a manual N-for-M trade against the user's live roster + league rules.

    ``give``/``get`` are lists of player_id (a lone string is tolerated). Reads the
    ``trade`` block baked into the snapshot payload and delegates the valuation to
    :func:`fantasy.decisions.trades.evaluate_trade_package`.
    """
    from fantasy.decisions.trades import evaluate_trade_package
    from fantasy.league_settings import LeagueSettings, RosterRequirements

    tb = payload.get("trade")
    if not tb or not tb.get("players"):
        return {"error": "Trade data isn't built for this league yet — build the analysis first."}

    give = [give] if isinstance(give, str) else list(give or [])
    get = [get] if isinstance(get, str) else list(get or [])
    give = [g for g in give if g]
    get = [h for h in get if h]
    if not give or not get:
        return {"error": "Add at least one player on each side of the trade."}
    if len(set(give)) != len(give) or len(set(get)) != len(get):
        return {"error": "The same player is listed twice on one side."}
    if set(give) & set(get):
        return {"error": "A player can't be on both sides of the trade."}

    players = tb["players"]
    teams = tb.get("teams", {})
    my_team_id = int(tb["my_team_id"])

    for pid in give + get:
        if pid not in players:
            return {"error": "One of the selected players isn't in this league."}
    for pid in give:
        if int(players[pid].get("team_id", -1)) != my_team_id:
            return {"error": f"{players[pid]['name']} isn't on your roster."}
    get_teams = {int(players[pid].get("team_id", -1)) for pid in get}
    if my_team_id in get_teams:
        return {"error": "You already own one of the players you're trying to acquire."}

    single = len(get_teams) == 1
    counter_roster = teams.get(str(next(iter(get_teams)))) if single else None

    rem = int(tb.get("remaining_weeks", 1)) or 1
    ros = {pid: p["proj"] * rem for pid, p in players.items()}
    ros_vor = {pid: p["vor"] * rem for pid, p in players.items()}
    pos = {pid: p["pos"] for pid, p in players.items()}
    names = {pid: p["name"] for pid, p in players.items()}
    league = LeagueSettings(team_count=tb.get("team_count", 12),
                            roster=RosterRequirements(slots=dict(tb.get("roster_slots", {}))))

    result = evaluate_trade_package(
        my_roster=teams.get(str(my_team_id), []), counter_roster=counter_roster,
        give=give, get=get, ros=ros, ros_vor=ros_vor, pos=pos, league=league,
        bench_size=int(tb.get("bench_size", 0)), ir_size=int(tb.get("ir_size", 0)),
        single_counterparty=single, names=names)
    if single:
        result["with_team"] = tb.get("team_names", {}).get(str(next(iter(get_teams))))
    return result
