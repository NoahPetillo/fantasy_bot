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
from pathlib import Path

from fastapi import Body, FastAPI, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from fantasy.config import ExecutionMode, settings
from fantasy.orchestrator.models import Proposal, ProposalStatus
from fantasy.orchestrator.store import Store

_STATIC = Path(__file__).resolve().parent / "static" / "dashboard.html"

log = logging.getLogger(__name__)

app = FastAPI(title="Fantasy App", version="0.1.0")
_store: Store | None = None


def store() -> Store:
    global _store
    if _store is None:
        _store = Store()
    return _store


def on_approved(p: Proposal) -> None:
    """Execution hook (Phase 3). In advise mode this is a no-op; in approve/auto
    mode it runs the swappable executor under the idempotency guard."""
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


@app.get("/api/dashboard")
def api_dashboard() -> dict:
    from fantasy.api.dashboard_data import read_snapshot

    payload = read_snapshot() or _fallback_payload()
    # Overlay live proposal statuses so approve/reject is reflected on refresh.
    for a in payload.get("actions", []):
        p = store().get(a.get("id", ""))
        if p:
            a["status"] = p.status.value
    return payload


@app.post("/api/analyze-trade")
def api_analyze_trade(body: dict = Body(...)) -> dict:
    from fantasy.api.dashboard_data import analyze_trade, read_snapshot

    snap = read_snapshot() or {}
    return analyze_trade(body.get("give", ""), body.get("get", ""),
                         snap.get("board_index", {}))
