"""Per-user, Postgres-backed proposal store.

Mirrors the legacy SQLite :class:`fantasy.orchestrator.store.Store` surface
(``add``/``get``/``by_key``/``has_executed``/``set_status``/``list``/
``merge_payload``) so the decision layer (``assemble``), the influence ledger, and
the confirm/approve endpoints work against it unchanged — but **every operation is
scoped to ``user_id``**, which is how per-user isolation (hard requirement #1) is
enforced at the query layer for proposals. Optionally also scoped to one
``league_id``.

The full domain :class:`~fantasy.orchestrator.models.Proposal` is stored in the
row's JSON ``payload``; the columns (``kind``/``status``/``value``/
``idempotency_key``/``league_id``) mirror the fields we query by and are kept in
sync on every mutation.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from fantasy.db.models import Proposal as ProposalRow
from fantasy.orchestrator.models import Proposal, ProposalStatus


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class PgProposalStore:
    def __init__(self, db: Session, user_id, league_id=None):
        self.db = db
        self.user_id = user_id
        self.league_id = league_id  # ORM league uuid (optional extra scope)

    # ── writes ──
    def add(self, p: Proposal) -> bool:
        """Insert a proposal for this user. Returns False if the (per-user)
        idempotency key already exists — the dedupe invariant."""
        if p.idempotency_key and self.by_key(p.idempotency_key) is not None:
            return False
        row = ProposalRow(
            id=p.id, user_id=self.user_id, league_id=self.league_id,
            kind=p.kind.value, status=p.status.value, value=float(p.value),
            idempotency_key=p.idempotency_key, payload=p.model_dump(mode="json"),
        )
        self.db.add(row)
        try:
            self.db.commit()
        except IntegrityError:
            self.db.rollback()  # concurrent insert of the same key
            return False
        return True

    def merge_payload(self, proposal_id: str, updates: dict) -> None:
        row = self._row(proposal_id)
        if row is None:
            return
        pay = dict(row.payload or {})
        inner = dict(pay.get("payload") or {})
        inner.update(updates)
        pay["payload"] = inner
        row.payload = pay  # reassign so the JSON column change is tracked
        self.db.commit()

    def set_status(self, proposal_id: str, status: ProposalStatus,
                   notify_ref: str | None = None) -> None:
        row = self._row(proposal_id)
        if row is None:
            return
        row.status = status.value
        pay = dict(row.payload or {})
        pay["status"] = status.value
        pay["updated_at"] = _now_iso()
        if notify_ref is not None:
            pay["notify_ref"] = notify_ref
        row.payload = pay
        self.db.commit()

    # ── reads (all user-scoped) ──
    def _row(self, proposal_id: str) -> ProposalRow | None:
        return self.db.execute(
            select(ProposalRow).where(ProposalRow.id == proposal_id,
                                      ProposalRow.user_id == self.user_id)
        ).scalar_one_or_none()

    def get(self, proposal_id: str) -> Proposal | None:
        row = self._row(proposal_id)
        return self._to_domain(row) if row else None

    def by_key(self, idempotency_key: str) -> Proposal | None:
        row = self.db.execute(
            select(ProposalRow).where(ProposalRow.idempotency_key == idempotency_key,
                                      ProposalRow.user_id == self.user_id)
        ).scalar_one_or_none()
        return self._to_domain(row) if row else None

    def has_executed(self, idempotency_key: str) -> bool:
        return self.db.execute(
            select(ProposalRow.id).where(
                ProposalRow.idempotency_key == idempotency_key,
                ProposalRow.user_id == self.user_id,
                ProposalRow.status == ProposalStatus.executed.value)
        ).first() is not None

    def list(self, status=None, season=None, week=None, kind=None, limit=200) -> list[Proposal]:
        stmt = select(ProposalRow).where(ProposalRow.user_id == self.user_id)
        if self.league_id is not None:
            stmt = stmt.where(ProposalRow.league_id == self.league_id)
        if status is not None:
            stmt = stmt.where(ProposalRow.status == getattr(status, "value", status))
        if kind is not None:
            stmt = stmt.where(ProposalRow.kind == getattr(kind, "value", kind))
        props = [self._to_domain(r) for r in self.db.execute(stmt).scalars().all()]
        # season/week live in the JSON payload — filter in Python (per-user sets are small).
        if season is not None:
            props = [p for p in props if p.season == season]
        if week is not None:
            props = [p for p in props if p.week == week]
        props.sort(key=lambda p: (p.value, p.created_at), reverse=True)
        return props[:limit]

    @staticmethod
    def _to_domain(row: ProposalRow) -> Proposal:
        return Proposal.model_validate(row.payload)
