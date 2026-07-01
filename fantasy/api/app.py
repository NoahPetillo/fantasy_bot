"""FastAPI service — the per-user approval surface + control panel.

Multi-tenant: every web endpoint is scoped to the Clerk-authenticated user, and
their leagues/snapshots/proposals live in Postgres (see fantasy/db). Auth is
Clerk-only (the shared-password gate was removed in Phase 4). The per-user approve
path is READ-ONLY to ESPN — it records the decision and never runs the execute
hook. The Slack integration keeps the legacy global store (owner single-tenant).
"""

from __future__ import annotations

import json
import logging
import threading
import uuid

from fastapi import Body, Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from sqlalchemy.orm import Session

from fantasy.api import ratelimit
from fantasy.api.clerk_auth import clerk_issuer, frontend_api_host, get_current_user
from fantasy.api.espn_routes import router as espn_router
from fantasy.api.user_build import build_full_for, build_shell_for
from fantasy.config import ExecutionMode, settings
from fantasy.db.base import get_db, get_sessionmaker
from fantasy.db.models import League, User
from fantasy.db.proposal_store import PgProposalStore
from fantasy.db.repos import (
    add_league,
    get_league,
    latest_snapshot,
    list_leagues,
    remove_league,
)
from fantasy.espn.client import EspnAuthError
from fantasy.espn.credentials import build_client_for_user
from fantasy.orchestrator.influence import influence_stats
from fantasy.orchestrator.models import Proposal, ProposalKind, ProposalStatus
from fantasy.orchestrator.store import Store
from pathlib import Path

_STATIC = Path(__file__).resolve().parent / "static" / "dashboard.html"
_CONNECT_STATIC = Path(__file__).resolve().parent / "static" / "connect.html"

log = logging.getLogger(__name__)

app = FastAPI(title="Fantasy App", version="0.1.0")
app.include_router(espn_router)  # per-user connect-ESPN + account endpoints
_store: Store | None = None
# league uuid (str) -> "building" | "done" | "error: ..."
_build_status: dict[str, str] = {}


@app.get("/api/config")
def api_config() -> dict:
    """Public bootstrap config for the frontend: the Clerk publishable key (used to
    initialize Clerk.js) and whether auth is configured on this server."""
    return {"clerk_publishable_key": settings.clerk_publishable_key,
            "clerk_frontend_api": frontend_api_host(),
            "auth_configured": bool(clerk_issuer())}


@app.get("/api/me")
def api_me(user: User = Depends(get_current_user)) -> dict:
    """The authenticated user (Clerk-verified). Provisions the ``users`` row on
    first login."""
    return {"id": str(user.id), "clerk_user_id": user.clerk_user_id,
            "email": user.email, "plan": user.plan}


# ── legacy global store + execution hook (Slack / owner single-tenant only) ────
def store() -> Store:
    global _store
    if _store is None:
        _store = Store()
    return _store


_chat_limiter: ratelimit.RateLimiter | None = None


def chat_limiter() -> ratelimit.RateLimiter:
    global _chat_limiter
    if _chat_limiter is None:
        _chat_limiter = ratelimit.RateLimiter(settings.chat_rate_limit,
                                              settings.chat_rate_window_seconds)
    return _chat_limiter


def on_approved(p: Proposal) -> None:
    """Legacy execution hook (Slack/owner path). Moments post to the group chat;
    ESPN actions stay gated by execution_mode. The multi-tenant web path does NOT
    call this — it is read-only to ESPN."""
    if p.kind == ProposalKind.moment:
        from fantasy.moments.publisher import publish_moment

        ref = publish_moment(p)
        if ref:
            store().set_status(p.id, ProposalStatus.executed, ref)
            log.info("Posted moment %s -> %s", p.id, ref)
        else:
            log.warning("Moment %s approved but post failed (check DISCORD_WEBHOOK_URL).", p.id)
        return
    if settings.execution_mode == ExecutionMode.advise:
        log.info("[advise] approved %s (%s) — no write performed.", p.id, p.title)
        return
    from fantasy.execute.base import execute_approved

    result = execute_approved(p, store())
    log.info("[%s/%s] approved %s -> %s: %s", settings.execution_mode.value,
             result.backend, p.id, "ok" if result.ok else "FAILED", result.message)


def _decide_legacy(proposal_id: str, approve: bool) -> dict:
    p = store().get(proposal_id)
    if p is None:
        raise HTTPException(404, "proposal not found")
    if p.status not in (ProposalStatus.proposed,):
        return {"id": p.id, "status": p.status.value, "note": "already decided"}
    new = ProposalStatus.approved if approve else ProposalStatus.rejected
    store().set_status(p.id, new)
    if approve:
        on_approved(p)
    return {"id": p.id, "status": new.value, "title": p.title}


@app.post("/slack/interactions")
async def slack_interactions(payload: str = Form(...)) -> dict:
    """Slack Block Kit button clicks — the owner's single-tenant channel (legacy)."""
    data = json.loads(payload)
    for action in data.get("actions", []):
        pid = action.get("value")
        if action.get("action_id") == "approve_proposal":
            return _decide_legacy(pid, True)
        if action.get("action_id") == "reject_proposal":
            return _decide_legacy(pid, False)
    return {"ok": True}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "mode": settings.execution_mode.value}


# ── per-user helpers ───────────────────────────────────────────────────────────
def _user_store(db: Session, user: User, league_id=None) -> PgProposalStore:
    return PgProposalStore(db, user.id, league_id)


def _league_dict(db: Session, lg: League) -> dict:
    built = latest_snapshot(db, lg.id) is not None
    return {"league_id": str(lg.id), "espn_league_id": lg.espn_league_id,
            "team_id": lg.team_id, "season": lg.season,
            "name": lg.name or f"League {lg.espn_league_id}", "built": built,
            "build_status": _build_status.get(str(lg.id), "done" if built else "shell")}


def _empty_dashboard() -> dict:
    return {
        "team": {"name": "No league yet", "league": "Add a league to get started",
                 "week": "—", "mode": settings.execution_mode.value},
        "kpis": [{"label": "Status", "value": "—", "sub": "no data"}],
        "waivers": [], "trades": [], "lineup": [], "lineup_total": 0, "standings": [],
        "feed": [], "actions": [], "board_index": {}, "statuses": {},
    }


# ── per-user proposals ─────────────────────────────────────────────────────────
@app.get("/api/proposals")
def api_proposals(status: str | None = None, season: int | None = None,
                  week: int | None = None, kind: str | None = None,
                  user: User = Depends(get_current_user),
                  db: Session = Depends(get_db)) -> list[dict]:
    st = ProposalStatus(status) if status else None
    props = _user_store(db, user).list(status=st, season=season, week=week, kind=kind)
    return [p.model_dump() for p in props]


def _decide_user(db: Session, user: User, proposal_id: str, approve: bool) -> dict:
    store_ = _user_store(db, user)
    p = store_.get(proposal_id)  # user-scoped: another user's id → None → 404
    if p is None:
        raise HTTPException(404, "proposal not found")
    if p.status not in (ProposalStatus.proposed,):
        return {"id": p.id, "status": p.status.value, "note": "already decided"}
    new = ProposalStatus.approved if approve else ProposalStatus.rejected
    store_.set_status(p.id, new)  # read-only to ESPN: no execute hook on this path
    return {"id": p.id, "status": new.value, "title": p.title}


@app.post("/proposals/{proposal_id}/approve")
def approve(proposal_id: str, user: User = Depends(get_current_user),
            db: Session = Depends(get_db)) -> dict:
    return _decide_user(db, user, proposal_id, True)


@app.post("/proposals/{proposal_id}/reject")
def reject(proposal_id: str, user: User = Depends(get_current_user),
           db: Session = Depends(get_db)) -> dict:
    return _decide_user(db, user, proposal_id, False)


@app.post("/proposals/{proposal_id}/undo")
def undo(proposal_id: str, user: User = Depends(get_current_user),
         db: Session = Depends(get_db)) -> dict:
    store_ = _user_store(db, user)
    p = store_.get(proposal_id)
    if p is None:
        raise HTTPException(404, "proposal not found")
    if p.status == ProposalStatus.executed:
        return {"id": p.id, "status": p.status.value, "note": "already executed — cannot undo"}
    store_.set_status(p.id, ProposalStatus.proposed)
    return {"id": p.id, "status": ProposalStatus.proposed.value}


@app.post("/proposals/{proposal_id}/confirm")
def confirm(proposal_id: str, user: User = Depends(get_current_user),
            db: Session = Depends(get_db)) -> dict:
    """Verify on ESPN (using THIS user's cookies) that an approved move landed."""
    store_ = _user_store(db, user)
    p = store_.get(proposal_id)
    if p is None:
        raise HTTPException(404, "proposal not found")
    lid = p.payload.get("league_id")
    if not lid:
        return {"id": p.id, "confirmed": False, "status": p.status.value,
                "detail": "No league recorded on this proposal — rebuild to enable verification."}
    try:
        client = build_client_for_user(db, user, int(lid), p.season)
    except EspnAuthError:
        return {"id": p.id, "confirmed": False, "status": p.status.value,
                "detail": "Connect your ESPN account to verify this move."}
    from fantasy.api.confirm import confirm_on_espn

    result = confirm_on_espn(p, client=client)
    if result.get("confirmed"):
        store_.set_status(p.id, ProposalStatus.executed)
        result["status"] = ProposalStatus.executed.value
    else:
        result["status"] = p.status.value
    return {"id": p.id, **result}


# ── per-user leagues ───────────────────────────────────────────────────────────
@app.get("/api/leagues")
def api_leagues(user: User = Depends(get_current_user),
                db: Session = Depends(get_db)) -> dict:
    leagues = [_league_dict(db, lg) for lg in list_leagues(db, user)]
    return {"leagues": leagues, "active": leagues[0]["league_id"] if leagues else None}


@app.post("/api/leagues")
def api_add_league(body: dict = Body(...), user: User = Depends(get_current_user),
                   db: Session = Depends(get_db)) -> dict:
    """Register one of the user's leagues, validate it against ESPN with their
    cookies, and write an instant shell snapshot. Full analysis builds on demand."""
    espn_id = body.get("league_id") or body.get("espn_league_id")
    try:
        espn_league_id = int(espn_id)
        team_id = int(body["team_id"]) if body.get("team_id") not in (None, "") else None
        season = int(body.get("season") or settings.espn_season)
    except (TypeError, ValueError):
        raise HTTPException(400, "league_id (and ideally team_id, season) required")

    lg = add_league(db, user, espn_league_id, team_id, season, name=body.get("name"))
    try:
        build_shell_for(db, user, lg)  # validates access with the user's cookies
    except EspnAuthError:
        remove_league(db, user, lg.id)
        raise HTTPException(400, "Connect your ESPN account first (Settings → Connect ESPN).")
    except Exception as e:  # noqa: BLE001
        remove_league(db, user, lg.id)
        raise HTTPException(400, f"Couldn't reach that league (check the id/team/season): {e}")
    return {"ok": True, "league": _league_dict(db, lg)}


@app.delete("/api/leagues/{league_id}")
def api_remove_league(league_id: str, user: User = Depends(get_current_user),
                      db: Session = Depends(get_db)) -> dict:
    if not remove_league(db, user, league_id):
        raise HTTPException(404, "league not found")
    return {"ok": True, "removed": league_id}


def _run_user_build(user_id: str, league_id: str, week: int | None) -> None:
    """Background full build in its own DB session (threads can't share one)."""
    db = get_sessionmaker()()
    try:
        user = db.get(User, uuid.UUID(user_id))
        lg = db.get(League, uuid.UUID(league_id))
        if user is None or lg is None or lg.user_id != user.id:
            _build_status[league_id] = "error: not found"
            return
        build_full_for(db, user, lg, week=week)
        _build_status[league_id] = "done"
    except EspnAuthError:
        _build_status[league_id] = "error: connect ESPN"
    except Exception as e:  # noqa: BLE001
        log.exception("user build failed for league %s", league_id)
        _build_status[league_id] = f"error: {e}"
    finally:
        db.close()


@app.post("/api/leagues/{league_id}/build")
def api_build_league(league_id: str, week: int | None = None,
                     user: User = Depends(get_current_user),
                     db: Session = Depends(get_db)) -> dict:
    lg = get_league(db, user, league_id)
    if lg is None:
        raise HTTPException(404, "league not found")
    key = str(lg.id)
    if _build_status.get(key) == "building":
        return {"ok": True, "status": "building", "note": "already in progress"}
    _build_status[key] = "building"
    threading.Thread(target=_run_user_build, args=(str(user.id), key, week), daemon=True).start()
    return {"ok": True, "status": "building"}


@app.get("/api/dashboard")
def api_dashboard(league: str | None = None, user: User = Depends(get_current_user),
                  db: Session = Depends(get_db)) -> dict:
    if league:  # explicit league must be one of the user's (else 404, no silent empty)
        lg = get_league(db, user, league)
        if lg is None:
            raise HTTPException(404, "league not found")
    else:
        leagues = list_leagues(db, user)
        if not leagues:
            return _empty_dashboard()
        lg = leagues[0]
    payload = latest_snapshot(db, lg.id) or _empty_dashboard()
    payload.setdefault("team", {})["build_status"] = _build_status.get(str(lg.id))
    store_ = _user_store(db, user, lg.id)
    t = payload.get("team", {})
    statuses = {pr.id: pr.status.value for pr in store_.list(season=t.get("season"), limit=500)}
    payload["statuses"] = statuses
    for a in payload.get("actions", []):
        if a.get("id") in statuses:
            a["status"] = statuses[a["id"]]
    if t.get("season") is not None:
        payload["influence"] = influence_stats(store_, season=t.get("season"),
                                               team_id=t.get("team_id"))
    return payload


@app.post("/api/chat")
def api_chat(request: Request, body: dict = Body(...),
             user: User = Depends(get_current_user),
             db: Session = Depends(get_db)) -> dict:
    """NFL/league Q&A over the user's own league snapshot (a logged-in feature).
    The per-IP limiter stays as an abuse floor; a per-user plan quota lands in
    Phase 5."""
    allowed, retry = chat_limiter().check(ratelimit.client_ip(request))
    if not allowed:
        mins = max(1, retry // 60)
        raise HTTPException(429, f"Too many questions — try again in about {mins} min.",
                            headers={"Retry-After": str(retry)})
    from fantasy.chat.agent import answer
    from fantasy.chat.tools import ChatContext

    question = (body.get("question") or "").strip()
    leagues = list_leagues(db, user)
    lg = get_league(db, user, body.get("league")) if body.get("league") else (leagues[0] if leagues else None)
    snap = (latest_snapshot(db, lg.id) if lg else {}) or {}
    ctx = ChatContext.from_snapshot(snap)
    return answer(question, ctx)


@app.post("/api/analyze-trade")
def api_analyze_trade(body: dict = Body(...), user: User = Depends(get_current_user),
                      db: Session = Depends(get_db)) -> dict:
    from fantasy.api.dashboard_data import analyze_trade

    leagues = list_leagues(db, user)
    lg = get_league(db, user, body.get("league")) if body.get("league") else (leagues[0] if leagues else None)
    snap = (latest_snapshot(db, lg.id) if lg else {}) or {}
    return analyze_trade(body.get("give", ""), body.get("get", ""), snap.get("board_index", {}))


@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse(_STATIC)


@app.get("/connect", response_class=HTMLResponse)
def connect_page():
    """The Connect-ESPN consent screen (shell; its API calls are Clerk-authenticated)."""
    return FileResponse(_CONNECT_STATIC)


@app.get("/privacy", response_class=HTMLResponse)
def privacy_page():
    from fantasy.legal import render_policy_html

    return HTMLResponse(render_policy_html("privacy"))


@app.get("/terms", response_class=HTMLResponse)
def terms_page():
    from fantasy.legal import render_policy_html

    return HTMLResponse(render_policy_html("terms"))
