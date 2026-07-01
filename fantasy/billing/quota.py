"""Per-user daily chat quota (hard requirement #5 — chat costs money per user).

Backed by the ``chat_usage`` table, scoped to (user, day), enforced by plan. This
is separate from the per-IP rate limiter, which stays as an anonymous-abuse floor.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from fantasy.billing.plans import chat_daily_limit
from fantasy.db.models import ChatUsage, User


def _today():
    return datetime.now(timezone.utc).date()


def chat_status(db: Session, user: User) -> dict:
    limit = chat_daily_limit(user.plan)
    row = db.get(ChatUsage, (user.id, _today()))
    used = row.count if row else 0
    return {"used": used, "limit": limit, "remaining": max(0, limit - used)}


def _locked_row(db: Session, user: User, day):
    """The (user, day) usage row, row-locked so concurrent chats can't both slip
    under the limit (no-op lock on SQLite, which serializes writes anyway)."""
    return db.execute(
        select(ChatUsage).where(ChatUsage.user_id == user.id, ChatUsage.day == day)
        .with_for_update()
    ).scalar_one_or_none()


def consume_chat(db: Session, user: User) -> bool:
    """Reserve one chat question for today (call BEFORE the LLM, which costs money).
    Returns False if the plan's daily limit is already reached — nothing consumed.
    Concurrency-safe: the usage row is locked for the check-and-increment."""
    limit = chat_daily_limit(user.plan)
    if limit <= 0:
        return False
    today = _today()
    row = _locked_row(db, user, today)
    if row is None:
        row = ChatUsage(user_id=user.id, day=today, count=0)
        db.add(row)
        try:
            db.flush()
        except IntegrityError:  # a concurrent request created it first — re-fetch locked
            db.rollback()
            row = _locked_row(db, user, today)
    if row.count >= limit:
        db.commit()  # release the lock
        return False
    row.count += 1
    db.commit()
    return True
