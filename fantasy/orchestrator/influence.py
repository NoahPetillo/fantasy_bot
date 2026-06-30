"""Bot-influence ledger — how much does the user actually follow the bot?

Reads the proposal store and summarizes, per league/season, which recommendations
the user FOLLOWED (approved/executed) vs. REJECTED vs. still pending. This is the
"are you listening to the bot, and is it helping?" tracker the dashboard surfaces
next to the retrospective Season Report Card (which shows what following the bot's
trades/start-sit *would* have realized).

Counts are computed live from the store so they refresh the instant a card is
approved/rejected/undone — no rebuild needed.
"""

from __future__ import annotations

from fantasy.orchestrator.models import ProposalKind, ProposalStatus
from fantasy.orchestrator.store import Store

ACTIONABLE = {ProposalKind.start_sit, ProposalKind.waiver, ProposalKind.trade}
FOLLOWED = {ProposalStatus.approved, ProposalStatus.executed}


def influence_stats(store: Store, season: int | None = None,
                    team_id: int | None = None, limit: int = 60) -> dict:
    props = store.list(season=season, limit=500)
    if team_id is not None:
        props = [p for p in props if p.team_id == team_id]
    act = [p for p in props if p.kind in ACTIONABLE]

    followed = [p for p in act if p.status in FOLLOWED]
    confirmed = [p for p in act if p.status == ProposalStatus.executed]
    rejected = [p for p in act if p.status == ProposalStatus.rejected]
    pending = [p for p in act if p.status == ProposalStatus.proposed]
    decided = len(followed) + len(rejected)
    rate = round(len(followed) / decided, 2) if decided else None

    log = [{
        "id": p.id, "kind": p.kind.value, "title": p.title,
        "status": p.status.value, "value": round(p.value, 1),
        "week": p.week, "decided_at": p.updated_at,
        "confirmed": p.status == ProposalStatus.executed,
    } for p in sorted(act, key=lambda x: x.updated_at, reverse=True)[:limit]]

    return {
        "followed": len(followed), "confirmed": len(confirmed),
        "rejected": len(rejected), "pending": len(pending),
        "decided": decided, "acceptance_rate": rate, "total": len(act),
        "by_kind": {k.value: len([p for p in act if p.kind == k]) for k in ACTIONABLE},
        "log": log,
    }
