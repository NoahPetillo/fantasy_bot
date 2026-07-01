"""Per-user ESPN credential service (encrypt at rest, decrypt only in memory).

This is the trust boundary for hard requirement #2: cookies are Fernet-encrypted
before they touch the database, decrypted only transiently to build an
:class:`~fantasy.espn.client.EspnClient` for that user's request, and never
logged, displayed, or returned. Deletion is immediate, and credentials auto-purge
after repeated ESPN auth failures.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from fantasy.db.models import EspnCredential, User
from fantasy.espn.account import brace_swid
from fantasy.espn.client import EspnAuthError, EspnClient
from fantasy.legal import ESPN_CONSENT_VERSION
from fantasy.security import crypto

log = logging.getLogger(__name__)

# Purge stored credentials after this many consecutive ESPN auth failures.
MAX_AUTH_FAILURES = 5

__all__ = [
    "ESPN_CONSENT_VERSION", "MAX_AUTH_FAILURES", "get_credential", "is_connected",
    "store_credentials", "get_decrypted_cookies", "delete_credentials",
    "record_auth_success", "record_auth_failure", "build_client_for_user",
]


def get_credential(db: Session, user: User) -> EspnCredential | None:
    return db.get(EspnCredential, user.id)


def is_connected(db: Session, user: User) -> bool:
    c = get_credential(db, user)
    return bool(c and c.status == "active")


def store_credentials(db: Session, user: User, espn_s2: str, swid: str,
                      consent_version: str) -> EspnCredential:
    """Encrypt + upsert this user's cookies. Consent version/time is persisted with
    them (the caller must have verified consent first). Resets failure state."""
    now = datetime.now(timezone.utc)
    s2_enc = crypto.encrypt(espn_s2.strip())
    swid_enc = crypto.encrypt(brace_swid(swid))  # store normalized (braced)
    c = get_credential(db, user)
    if c is None:
        c = EspnCredential(user_id=user.id, s2_enc=s2_enc, swid_enc=swid_enc,
                           status="active", failure_count=0,
                           consent_version=consent_version, consent_at=now)
        db.add(c)
    else:
        c.s2_enc, c.swid_enc = s2_enc, swid_enc
        c.status, c.failure_count = "active", 0
        c.consent_version, c.consent_at = consent_version, now
    db.commit()
    db.refresh(c)
    log.info("Stored ESPN credentials for user %s (consent %s)", user.id, consent_version)
    return c


def get_decrypted_cookies(db: Session, user: User) -> tuple[str, str] | None:
    """(espn_s2, swid) decrypted in memory, or None if not connected. The plaintext
    must never be logged or returned to the client."""
    c = get_credential(db, user)
    if not c or c.status != "active":
        return None
    return crypto.decrypt(c.s2_enc), crypto.decrypt(c.swid_enc)


def delete_credentials(db: Session, user: User) -> bool:
    """Immediately remove this user's stored cookies. Returns True if any existed."""
    c = get_credential(db, user)
    if not c:
        return False
    db.delete(c)
    db.commit()
    log.info("Deleted ESPN credentials for user %s", user.id)
    return True


def record_auth_success(db: Session, user: User) -> None:
    c = get_credential(db, user)
    if c and (c.failure_count or c.status != "active"):
        c.failure_count, c.status = 0, "active"
        db.commit()


def record_auth_failure(db: Session, user: User) -> bool:
    """Increment the failure counter; auto-purge past the threshold. Returns True
    if the credentials were purged.

    Locks the row (``with_for_update``) so two concurrent failing requests can't
    both increment/purge the same credential (no-op on SQLite, which serializes
    writes anyway)."""
    c = db.get(EspnCredential, user.id, with_for_update=True)
    if not c:
        return False
    c.failure_count += 1
    if c.failure_count >= MAX_AUTH_FAILURES:
        db.delete(c)
        db.commit()
        log.warning("Purged ESPN credentials for user %s after %d auth failures",
                    user.id, MAX_AUTH_FAILURES)
        return True
    db.commit()
    return False


def build_client_for_user(db: Session, user: User, league_id: int,
                          season: int | None = None) -> EspnClient:
    """Construct an EspnClient from THIS user's decrypted cookies (never global
    settings). Raises :class:`EspnAuthError` if the user hasn't connected ESPN."""
    cookies = get_decrypted_cookies(db, user)
    if cookies is None:
        raise EspnAuthError("No ESPN credentials connected for this user.")
    s2, swid = cookies
    return EspnClient(league_id=league_id, season=season, espn_s2=s2, swid=swid)
