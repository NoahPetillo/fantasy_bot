"""Per-user isolation at the data layer (hard requirement #1).

Proves at the schema/query level that one user's rows cannot collide with or be
read/mutated as another user's: FK + unique constraints, user-scoped queries, and
cascade delete. Endpoint-level isolation across every route is added in Phase 3;
this locks the foundation those queries rest on.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from fantasy.db.models import ChatUsage, EspnCredential, League, Proposal, Snapshot, User


def _user(db, clerk_id):
    u = User(clerk_user_id=clerk_id, email=f"{clerk_id}@ex.com")
    db.add(u)
    db.commit()
    db.refresh(u)
    return u


def test_two_users_isolated_rows(db):
    a, b = _user(db, "user_a"), _user(db, "user_b")
    # Same ESPN league id + season for BOTH users is allowed (scoped per user).
    la = League(user_id=a.id, espn_league_id=111, season=2025, name="A league")
    lb = League(user_id=b.id, espn_league_id=111, season=2025, name="B league")
    db.add_all([la, lb])
    db.commit()

    a_leagues = db.execute(select(League).where(League.user_id == a.id)).scalars().all()
    assert [l.id for l in a_leagues] == [la.id]  # A sees only A's league

    # B cannot fetch A's league by id when the query is scoped to B (the pattern
    # every per-user endpoint must use).
    stolen = db.execute(
        select(League).where(League.id == la.id, League.user_id == b.id)
    ).scalar_one_or_none()
    assert stolen is None


def test_league_unique_per_user(db):
    a = _user(db, "user_a")
    db.add(League(user_id=a.id, espn_league_id=222, season=2025))
    db.commit()
    db.add(League(user_id=a.id, espn_league_id=222, season=2025))  # dup for same user
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()


def test_proposal_idempotency_key_unique_per_user(db):
    a, b = _user(db, "user_a"), _user(db, "user_b")
    db.add(Proposal(id="pa1", user_id=a.id, kind="waiver", idempotency_key="k1"))
    db.add(Proposal(id="pb1", user_id=b.id, kind="waiver", idempotency_key="k1"))  # ok: different user
    db.commit()
    db.add(Proposal(id="pa2", user_id=a.id, kind="waiver", idempotency_key="k1"))  # dup key for A
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()


def test_foreign_key_enforced(db):
    import uuid

    # A league owned by a non-existent user must be rejected by the FK.
    db.add(League(user_id=uuid.uuid4(), espn_league_id=999, season=2025))
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()


def test_snapshot_read_isolation_via_league(db):
    """Snapshots are owned transitively through their league; the per-user query
    pattern (join snapshots → leagues → user) must never surface another user's."""
    a, b = _user(db, "user_a"), _user(db, "user_b")
    la = League(user_id=a.id, espn_league_id=1, season=2025)
    lb = League(user_id=b.id, espn_league_id=2, season=2025)
    db.add_all([la, lb])
    db.commit()
    db.add(Snapshot(league_id=la.id, week=1, payload={"secret": "A-only"}))
    db.commit()

    b_snaps = db.execute(
        select(Snapshot).join(League).where(League.user_id == b.id)
    ).scalars().all()
    assert b_snaps == []  # B sees none of A's snapshots

    a_snaps = db.execute(
        select(Snapshot).join(League).where(League.user_id == a.id)
    ).scalars().all()
    assert len(a_snaps) == 1 and a_snaps[0].payload["secret"] == "A-only"


def test_delete_account_cascades_only_that_user(db):
    a, b = _user(db, "user_a"), _user(db, "user_b")
    la = League(user_id=a.id, espn_league_id=1, season=2025)
    lb = League(user_id=b.id, espn_league_id=2, season=2025)
    db.add_all([la, lb])
    db.commit()
    db.add_all([
        EspnCredential(user_id=a.id, s2_enc="x", swid_enc="y", consent_version="v1",
                       consent_at=datetime.now(timezone.utc)),
        Snapshot(league_id=la.id, week=1, payload={"ok": True}),
        Proposal(id="pca", user_id=a.id, league_id=la.id, kind="trade", idempotency_key="ka"),
        ChatUsage(user_id=a.id, day=datetime.now(timezone.utc).date(), count=3),
        Proposal(id="pcb", user_id=b.id, league_id=lb.id, kind="trade", idempotency_key="kb"),
    ])
    db.commit()

    db.delete(a)  # "delete my account"
    db.commit()

    # A's data is gone…
    assert db.get(User, a.id) is None
    assert db.execute(select(League).where(League.user_id == a.id)).scalars().all() == []
    assert db.get(EspnCredential, a.id) is None
    assert db.execute(select(Snapshot).where(Snapshot.league_id == la.id)).scalars().all() == []
    assert db.execute(select(Proposal).where(Proposal.user_id == a.id)).scalars().all() == []
    # …B's data is untouched.
    assert db.get(User, b.id) is not None
    assert len(db.execute(select(Proposal).where(Proposal.user_id == b.id)).scalars().all()) == 1
