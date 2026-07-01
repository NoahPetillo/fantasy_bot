"""Verify on ESPN that an approved move actually landed on the user's roster.

In advise mode the bot never writes to ESPN — the user makes the move themselves.
This re-reads their live roster and checks the proposal's intended end-state is now
present (added player on / dropped player off, trade pieces swapped), so an approval
can be confirmed against reality before it counts toward the influence ledger.
"""

from __future__ import annotations

import logging

from fantasy.espn.client import EspnClient
from fantasy.orchestrator.models import Proposal, ProposalKind

log = logging.getLogger(__name__)


def _my_roster_ids(client: EspnClient, team_id: int | None) -> dict[str, str]:
    """gsis-or-espn id -> player name for the given team's current roster."""
    from fantasy.data.nfl import load_player_ids

    xwalk = load_player_ids()
    cols = {c.lower(): c for c in xwalk.columns}
    e_c, g_c = cols.get("espn_id"), cols.get("gsis_id")
    e2g = {}
    if e_c and g_c:
        m = xwalk[[e_c, g_c]].dropna()
        e2g = {str(int(e)): g for e, g in zip(m[e_c], m[g_c]) if str(e).strip()}
    out = {}
    for t in client.teams():
        if team_id is not None and getattr(t, "team_id", None) != team_id:
            continue
        for p in getattr(t, "roster", []) or []:
            eid = str(getattr(p, "playerId", "") or "")
            out[e2g.get(eid, f"espn:{eid}")] = getattr(p, "name", "?")
    return out


def confirm_on_espn(p: Proposal, client: EspnClient | None = None) -> dict:
    """Verify the proposal's end-state on ESPN. ``client`` should be built from the
    owning user's decrypted cookies (multi-tenant); if omitted it falls back to the
    legacy global client (single-tenant callers only)."""
    lid = p.payload.get("league_id")
    if not lid:
        return {"confirmed": False,
                "detail": "No league recorded on this proposal — rebuild the dashboard to enable verification."}
    try:
        if client is None:
            client = EspnClient(league_id=int(lid), season=p.season)
        roster = _my_roster_ids(client, p.team_id)
    except Exception as e:  # noqa: BLE001
        log.warning("confirm read failed: %s", e)
        return {"confirmed": False, "detail": f"Couldn't read ESPN roster: {e}"}

    def nm(pid):
        return roster.get(pid) or p.payload.get("names", {}).get(pid, pid)

    pay = p.payload
    if p.kind == ProposalKind.waiver:
        add, drop = pay.get("add"), pay.get("drop")
        on = add in roster
        off = drop not in roster
        ok = on and (off or not drop)
        return {"confirmed": ok,
                "detail": f"{'✓' if on else '✗'} {nm(add)} {'is on' if on else 'NOT on'} your roster"
                          + (f"; {'✓' if off else '✗'} {nm(drop)} {'dropped' if off else 'STILL rostered'}." if drop else ".")}
    if p.kind == ProposalKind.trade:
        get, give = pay.get("get"), pay.get("give")
        on = get in roster
        off = give not in roster
        ok = on and off
        return {"confirmed": ok,
                "detail": f"{'✓' if on else '✗'} {nm(get)} {'received' if on else 'NOT yet received'}; "
                          f"{'✓' if off else '✗'} {nm(give)} {'sent' if off else 'STILL on your roster'}."}
    if p.kind == ProposalKind.start_sit:
        starters = pay.get("key_fields", {}).get("starters", []) or []
        present = [s for s in starters if s in roster]
        ok = bool(starters) and len(present) == len(starters)
        return {"confirmed": ok,
                "detail": f"{len(present)}/{len(starters)} recommended starters are on your roster."
                          " (Lineup slotting itself isn't readable post-lock.)"}
    return {"confirmed": False, "detail": "Nothing to verify for this kind of proposal."}
