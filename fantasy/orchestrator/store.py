"""SQLite-backed append-only store for proposals (the action log).

Idempotency is enforced by a UNIQUE constraint on ``idempotency_key``: re-running
a cycle inserts new proposals but silently ignores ones already seen. Status
transitions (approve/reject/execute) are the only mutations. Before any Phase-3
write, callers MUST check there is no existing ``executed`` row for the key.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from pathlib import Path

from fantasy.config import settings
from fantasy.orchestrator.models import Proposal, ProposalStatus

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS proposals (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    team_id INTEGER,
    title TEXT NOT NULL,
    detail TEXT,
    payload TEXT,
    value REAL,
    confidence REAL,
    status TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    notify_ref TEXT,
    created_at TEXT,
    updated_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_prop_status ON proposals(status);
CREATE INDEX IF NOT EXISTS ix_prop_week ON proposals(season, week);
"""


class Store:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else settings.db_path
        # check_same_thread=False: the scheduler and the API both touch the store;
        # a lock serializes writes since one sqlite connection isn't thread-safe.
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    # ── writes ──
    def add(self, p: Proposal) -> bool:
        """Insert a proposal. Returns True if new, False if a duplicate key existed."""
        try:
            with self._lock:
                self.conn.execute(
                    """INSERT INTO proposals
                       (id, kind, season, week, team_id, title, detail, payload, value,
                        confidence, status, idempotency_key, notify_ref, created_at, updated_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (p.id, p.kind.value, p.season, p.week, p.team_id, p.title, p.detail,
                     json.dumps(p.payload), p.value, p.confidence, p.status.value,
                     p.idempotency_key, p.notify_ref, p.created_at, p.updated_at),
                )
                self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False  # idempotency_key already present

    def set_status(self, proposal_id: str, status: ProposalStatus, notify_ref: str | None = None):
        from fantasy.orchestrator.models import _now

        with self._lock:
            self.conn.execute(
                "UPDATE proposals SET status=?, updated_at=?, notify_ref=COALESCE(?, notify_ref) WHERE id=?",
                (status.value, _now(), notify_ref, proposal_id),
            )
            self.conn.commit()

    # ── reads ──
    def _row_to_proposal(self, r: sqlite3.Row) -> Proposal:
        d = dict(r)
        d["payload"] = json.loads(d["payload"] or "{}")
        return Proposal.model_validate(d)

    def get(self, proposal_id: str) -> Proposal | None:
        r = self.conn.execute("SELECT * FROM proposals WHERE id=?", (proposal_id,)).fetchone()
        return self._row_to_proposal(r) if r else None

    def by_key(self, idempotency_key: str) -> Proposal | None:
        r = self.conn.execute(
            "SELECT * FROM proposals WHERE idempotency_key=?", (idempotency_key,)
        ).fetchone()
        return self._row_to_proposal(r) if r else None

    def has_executed(self, idempotency_key: str) -> bool:
        r = self.conn.execute(
            "SELECT 1 FROM proposals WHERE idempotency_key=? AND status=? LIMIT 1",
            (idempotency_key, ProposalStatus.executed.value),
        ).fetchone()
        return r is not None

    def list(self, status=None, season=None, week=None, kind=None, limit=200) -> list[Proposal]:
        q, args = "SELECT * FROM proposals WHERE 1=1", []
        for col, val in [("status", status), ("season", season), ("week", week), ("kind", kind)]:
            if val is not None:
                q += f" AND {col}=?"
                args.append(val.value if hasattr(val, "value") else val)
        q += " ORDER BY value DESC, created_at DESC LIMIT ?"
        args.append(limit)
        return [self._row_to_proposal(r) for r in self.conn.execute(q, args).fetchall()]

    def close(self):
        self.conn.close()
