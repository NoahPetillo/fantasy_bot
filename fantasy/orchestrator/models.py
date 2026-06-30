"""Proposal — one recommended action awaiting human approve/reject.

Every recommendation the system makes (start/sit, waiver claim, trade) becomes a
Proposal with a stable ``idempotency_key`` so the same advice is never logged or
notified twice, and (in Phase 3) never executed twice. This is the append-only
action log the architecture mandates.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field


class ProposalKind(str, Enum):
    start_sit = "start_sit"
    waiver = "waiver"
    trade = "trade"
    alert = "alert"  # news/injury heads-up, no action to approve
    moment = "moment"  # league hype content awaiting approve-to-post (content engine)


class ProposalStatus(str, Enum):
    proposed = "proposed"
    approved = "approved"
    rejected = "rejected"
    executed = "executed"
    expired = "expired"
    failed = "failed"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Proposal(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    kind: ProposalKind
    season: int
    week: int
    team_id: int | None = None
    title: str
    detail: str = ""
    payload: dict = Field(default_factory=dict)
    value: float = 0.0  # ranking score: expected points/VOR gain
    confidence: float = 0.5
    status: ProposalStatus = ProposalStatus.proposed
    idempotency_key: str = ""
    notify_ref: str | None = None
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)

    def model_post_init(self, _ctx) -> None:
        if not self.idempotency_key:
            self.idempotency_key = self.compute_key()

    def compute_key(self) -> str:
        """Stable hash over the action's identity (NOT its rationale/value).

        Two runs that produce the same underlying move in the same week collapse
        to one proposal. ``payload['key_fields']`` (if present) defines identity;
        otherwise the whole payload is used.
        """
        ident = self.payload.get("key_fields", self.payload)
        blob = json.dumps(
            {"kind": self.kind.value, "season": self.season, "week": self.week,
             "team": self.team_id, "ident": ident},
            sort_keys=True, default=str,
        )
        return hashlib.sha1(blob.encode()).hexdigest()[:16]
