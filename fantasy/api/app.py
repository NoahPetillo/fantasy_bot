"""FastAPI service — the approval surface + control panel.

In Phase 2 (``advise`` mode) approving a proposal just records the decision. In
Phase 3, ``on_approved`` becomes the execution hook (set lineup / submit waiver /
propose trade) — gated by the idempotency log so nothing executes twice.

Endpoints:
- GET  /health
- GET  /api/proposals
- POST /proposals/{id}/approve | /reject
- POST /slack/interactions          (Slack button clicks)
- GET  /                            (minimal list; the polished dashboard is Phase 5)
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from fastapi import Body, Depends, FastAPI, Form, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from fantasy.api import auth, ratelimit
from fantasy.api.clerk_auth import get_current_user
from fantasy.api.espn_routes import router as espn_router
from fantasy.config import ExecutionMode, settings
from fantasy.db.models import User
from fantasy.leagues import LeagueRef, registry
from fantasy.orchestrator.models import Proposal, ProposalKind, ProposalStatus
from fantasy.orchestrator.store import Store

_STATIC = Path(__file__).resolve().parent / "static" / "dashboard.html"
_CONNECT_STATIC = Path(__file__).resolve().parent / "static" / "connect.html"

log = logging.getLogger(__name__)

app = FastAPI(title="Fantasy App", version="0.1.0")
app.include_router(espn_router)  # per-user connect-ESPN + account endpoints
_store: Store | None = None
# league_id (str) -> "building" | "done" | "error: ..."
_build_status: dict[str, str] = {}

registry().seed_default()  # bootstrap the .env league into the registry on first run


@app.middleware("http")
async def _auth_gate(request: Request, call_next):
    """Shared-password gate. When a password is configured, every path except the
    public allowlist (chatbot, login, health, static shell) requires a valid
    session cookie; otherwise it's a no-op. See fantasy/api/auth.py."""
    if auth.gate_enabled() and not auth.is_public(request.url.path):
        if not auth.valid_token(request.cookies.get(auth.COOKIE)):
            return JSONResponse({"detail": "password required"}, status_code=401)
    return await call_next(request)


@app.get("/api/session")
def api_session(request: Request) -> dict:
    """Tells the frontend whether a password is required and whether this browser
    is already unlocked — so it can show the lock screen or the full dashboard."""
    authed = not auth.gate_enabled() or auth.valid_token(request.cookies.get(auth.COOKIE))
    return {"gate_enabled": auth.gate_enabled(), "authed": authed}


@app.post("/api/login")
def api_login(request: Request, response: Response, body: dict = Body(...)) -> dict:
    """Exchange the shared password for a signed session cookie."""
    if not auth.gate_enabled():
        return {"ok": True, "note": "no password configured — site is open"}
    if not auth.check_password(str(body.get("password") or "")):
        raise HTTPException(401, "incorrect password")
    # Secure cookie only over https (so it still works on plain-http localhost).
    secure = request.headers.get("x-forwarded-proto", request.url.scheme) == "https"
    response.set_cookie(auth.COOKIE, auth.issue_token(), max_age=auth.TTL,
                        httponly=True, samesite="lax", secure=secure)
    return {"ok": True}


@app.post("/api/logout")
def api_logout(response: Response) -> dict:
    response.delete_cookie(auth.COOKIE)
    return {"ok": True}


@app.get("/api/me")
def api_me(user: User = Depends(get_current_user)) -> dict:
    """The authenticated user (Clerk-verified). Provisions the ``users`` row on
    first login. This is the first route on the new per-user auth path."""
    return {"id": str(user.id), "clerk_user_id": user.clerk_user_id,
            "email": user.email, "plan": user.plan}


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
    """Execution hook. Moments post to the group chat on approval (not an ESPN
    write, so it runs in any mode). ESPN actions stay gated by execution_mode:
    a no-op in advise mode, the swappable executor otherwise."""
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


@app.get("/health")
def health() -> dict:
    pend = len(store().list(status=ProposalStatus.proposed))
    return {"status": "ok", "mode": settings.execution_mode.value, "pending_proposals": pend}


@app.get("/api/proposals")
def api_proposals(status: str | None = None, season: int | None = None,
                  week: int | None = None, kind: str | None = None) -> list[dict]:
    st = ProposalStatus(status) if status else None
    return [p.model_dump() for p in store().list(status=st, season=season, week=week, kind=kind)]


def _decide(proposal_id: str, approve: bool) -> dict:
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


@app.post("/proposals/{proposal_id}/approve")
def approve(proposal_id: str) -> dict:
    return _decide(proposal_id, True)


@app.post("/proposals/{proposal_id}/reject")
def reject(proposal_id: str) -> dict:
    return _decide(proposal_id, False)


@app.post("/proposals/{proposal_id}/undo")
def undo(proposal_id: str) -> dict:
    """Revert a just-made decision back to 'proposed' (the dashboard's Undo toast).
    Refuses once an action has actually executed/posted (can't unsend)."""
    p = store().get(proposal_id)
    if p is None:
        raise HTTPException(404, "proposal not found")
    if p.status == ProposalStatus.executed:
        return {"id": p.id, "status": p.status.value, "note": "already executed — cannot undo"}
    store().set_status(p.id, ProposalStatus.proposed)
    return {"id": p.id, "status": ProposalStatus.proposed.value}


@app.post("/proposals/{proposal_id}/confirm")
def confirm(proposal_id: str) -> dict:
    """Verify on ESPN that an approved waiver/lineup move actually landed on your
    roster (you make the move in ESPN; this re-reads and confirms it). Marks the
    proposal 'executed' once the intended end-state is present."""
    p = store().get(proposal_id)
    if p is None:
        raise HTTPException(404, "proposal not found")
    from fantasy.api.confirm import confirm_on_espn

    result = confirm_on_espn(p)
    if result.get("confirmed"):
        store().set_status(p.id, ProposalStatus.executed)
        result["status"] = ProposalStatus.executed.value
    else:
        result["status"] = p.status.value
    return {"id": p.id, **result}


@app.post("/slack/interactions")
async def slack_interactions(payload: str = Form(...)) -> dict:
    """Handle Slack Block Kit button clicks (action_id approve_proposal/reject_proposal)."""
    data = json.loads(payload)
    for action in data.get("actions", []):
        pid = action.get("value")
        if action.get("action_id") == "approve_proposal":
            return _decide(pid, True)
        if action.get("action_id") == "reject_proposal":
            return _decide(pid, False)
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse(_STATIC)


@app.get("/connect", response_class=HTMLResponse)
def connect_page():
    """The Connect-ESPN consent screen (shell only; the API calls it makes are
    Clerk-authenticated). Full Clerk sign-in wraps this in Phase 4."""
    return FileResponse(_CONNECT_STATIC)


def _fallback_payload() -> dict:
    return {
        "team": {"name": "No snapshot yet", "league": "Run scripts/dashboard.py to populate",
                 "week": "—", "mode": settings.execution_mode.value},
        "kpis": [{"label": "Status", "value": "—", "sub": "no data"}],
        "waivers": [], "trades": [], "lineup": [], "lineup_total": 0, "standings": [],
        "feed": [], "actions": [{"id": p.id, "kind": p.kind.value, "title": p.title,
                                 "value": round(p.value, 1), "status": p.status.value}
                                for p in store().list(limit=40)],
        "board_index": {},
    }


# ── multi-league management ──────────────────────────────────────────────────
def _default_league_id() -> int | None:
    leagues = registry().all()
    return leagues[0].league_id if leagues else settings.espn_league_id


@app.get("/api/leagues")
def api_leagues() -> dict:
    from fantasy.api.dashboard_data import snapshot_path

    out = []
    for r in registry().all():
        built = snapshot_path(r.league_id).exists()
        out.append({"league_id": r.league_id, "team_id": r.team_id, "season": r.season,
                    "name": r.name or f"League {r.league_id}", "built": built,
                    "build_status": _build_status.get(str(r.league_id), "done" if built else "shell")})
    return {"leagues": out, "active": _default_league_id()}


@app.post("/api/leagues")
def api_add_league(body: dict = Body(...)) -> dict:
    """Register a league (id + team id + season). Validates against ESPN and writes
    an instant shell snapshot so it shows up immediately. Heavy analysis is built
    on demand via /api/leagues/{id}/build."""
    try:
        league_id = int(body["league_id"])
        team_id = int(body["team_id"]) if body.get("team_id") not in (None, "") else None
        season = int(body.get("season") or settings.espn_season)
    except (KeyError, ValueError, TypeError):
        raise HTTPException(400, "league_id (and ideally team_id, season) required")

    from fantasy.api.build import build_shell

    ref = registry().add(LeagueRef(league_id=league_id, team_id=team_id, season=season))
    try:
        build_shell(ref)  # validates cookies/access + records the league name
    except Exception as e:  # noqa: BLE001
        registry().remove(league_id)
        raise HTTPException(400, f"Couldn't reach that league (check the ID/cookies): {e}")
    return {"ok": True, "league": next((d for d in api_leagues()["leagues"]
                                        if d["league_id"] == league_id), None)}


@app.delete("/api/leagues/{league_id}")
def api_remove_league(league_id: int) -> dict:
    ok = registry().remove(league_id)
    if not ok:
        raise HTTPException(404, "league not registered")
    return {"ok": True, "removed": league_id}


def _run_build(league_id: int, week: int | None) -> None:
    from fantasy.api.build import build_full

    ref = registry().get(league_id)
    if ref is None:
        _build_status[str(league_id)] = "error: not registered"
        return
    _build_status[str(league_id)] = "building"
    try:
        build_full(ref, week=week)
        _build_status[str(league_id)] = "done"
    except Exception as e:  # noqa: BLE001
        log.exception("build failed for %s", league_id)
        _build_status[str(league_id)] = f"error: {e}"


@app.post("/api/leagues/{league_id}/build")
def api_build_league(league_id: int, week: int | None = None) -> dict:
    if registry().get(league_id) is None:
        raise HTTPException(404, "league not registered")
    if _build_status.get(str(league_id)) == "building":
        return {"ok": True, "status": "building", "note": "already in progress"}
    threading.Thread(target=_run_build, args=(league_id, week), daemon=True).start()
    return {"ok": True, "status": "building"}


@app.get("/api/dashboard")
def api_dashboard(league: int | None = None) -> dict:
    from fantasy.api.dashboard_data import read_snapshot
    from fantasy.orchestrator.influence import influence_stats

    league_id = league if league is not None else _default_league_id()
    payload = read_snapshot(league_id) or _fallback_payload()
    payload.setdefault("team", {})["build_status"] = _build_status.get(str(league_id), None)
    # Live status map (id -> status) so every card reflects approve/reject/undo/confirm.
    t = payload.get("team", {})
    statuses = {pr.id: pr.status.value for pr in store().list(season=t.get("season"), limit=500)}
    payload["statuses"] = statuses
    for a in payload.get("actions", []):
        if a.get("id") in statuses:
            a["status"] = statuses[a["id"]]
    # Recompute the influence ledger live so counts update the instant a card is decided.
    if t.get("season") is not None:
        payload["influence"] = influence_stats(store(), season=t.get("season"),
                                               team_id=t.get("team_id"))
    return payload


@app.post("/api/chat")
def api_chat(request: Request, body: dict = Body(...)) -> dict:
    """NFL/league Q&A. Routes the question to deterministic data tools (real
    nflverse stats + our projections + league scoring); the model only phrases.

    Public so league mates can use it without the password — but anonymous callers
    are rate-limited per IP. The authenticated owner (valid session cookie) is
    exempt."""
    if not auth.valid_token(request.cookies.get(auth.COOKIE)):
        allowed, retry = chat_limiter().check(ratelimit.client_ip(request))
        if not allowed:
            mins = max(1, retry // 60)
            raise HTTPException(429, f"Too many questions — try again in about {mins} min.",
                                headers={"Retry-After": str(retry)})
    from fantasy.api.dashboard_data import read_snapshot
    from fantasy.chat.agent import answer
    from fantasy.chat.tools import ChatContext

    question = (body.get("question") or "").strip()
    league = body.get("league")
    league_id = league if league is not None else _default_league_id()
    snap = read_snapshot(league_id) or {}
    ctx = ChatContext.from_snapshot(snap)
    return answer(question, ctx)


@app.post("/api/analyze-trade")
def api_analyze_trade(body: dict = Body(...)) -> dict:
    from fantasy.api.dashboard_data import analyze_trade, read_snapshot

    snap = read_snapshot() or {}
    return analyze_trade(body.get("give", ""), body.get("get", ""),
                         snap.get("board_index", {}))
