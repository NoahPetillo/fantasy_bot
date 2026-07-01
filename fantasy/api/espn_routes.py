"""Connect-ESPN + account endpoints (per-user, Clerk-authenticated).

Flow (Phase 2): show the consent copy → user checks the required box + pastes their
``espn_s2``/``SWID`` → we validate against ESPN, encrypt, and store, persisting the
consent version/time. Plus "Test connection", delete-credentials, and
delete-account. Responses NEVER include the cookie values.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Body, Depends, HTTPException
from sqlalchemy.orm import Session

from fantasy.api.clerk_auth import get_current_user
from fantasy.db.base import get_db
from fantasy.db.models import User
from fantasy.espn import credentials as creds
from fantasy.espn.account import discover_ff_leagues, validate_cookies
from fantasy.espn.client import EspnAuthError
from fantasy.legal import ESPN_CONSENT_VERSION, espn_consent_markdown

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["espn"])


@router.get("/legal/espn-consent")
def get_espn_consent() -> dict:
    """The consent copy + version to render at the connect step (public: it's the
    disclosure itself, shown before anything is stored)."""
    return {"version": ESPN_CONSENT_VERSION, "markdown": espn_consent_markdown()}


@router.get("/espn/status")
def espn_status(user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    c = creds.get_credential(db, user)
    if not c:
        return {"connected": False}
    return {
        "connected": c.status == "active",
        "status": c.status,
        "consent_version": c.consent_version,
        "consent_at": c.consent_at.isoformat() if c.consent_at else None,
        "updated_at": c.updated_at.isoformat() if c.updated_at else None,
        "failure_count": c.failure_count,
    }


def _discover_safe(s2: str, swid: str) -> list[dict]:
    try:
        return discover_ff_leagues(s2, swid)
    except Exception:  # noqa: BLE001 — discovery is best-effort, never fatal
        return []


@router.post("/espn/connect")
def espn_connect(body: dict = Body(...), user: User = Depends(get_current_user),
                 db: Session = Depends(get_db)) -> dict:
    """Store the user's ESPN cookies after they consent. Validates against ESPN
    first so we never persist dead credentials."""
    if not body.get("consent"):
        raise HTTPException(400, "You must accept the consent notice to connect ESPN.")
    espn_s2 = (body.get("espn_s2") or "").strip()
    swid = (body.get("swid") or "").strip()
    if not espn_s2 or not swid:
        raise HTTPException(400, "Both espn_s2 and SWID are required.")

    try:
        ok = validate_cookies(espn_s2, swid)
    except Exception:  # noqa: BLE001 — transient/unknown ESPN error, not an auth failure
        raise HTTPException(502, "Couldn't reach ESPN to verify your cookies. Please try again.")
    if not ok:
        raise HTTPException(400, "Those ESPN cookies didn't authenticate. Re-copy espn_s2 and SWID while logged in to ESPN.")

    creds.store_credentials(db, user, espn_s2, swid,
                            consent_version=body.get("consent_version") or ESPN_CONSENT_VERSION)
    return {"ok": True, "connected": True, "leagues_found": _discover_safe(espn_s2, swid)}


@router.post("/espn/test")
def espn_test(user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    """Re-validate the stored cookies against ESPN. On auth failure, count it and
    auto-purge past the threshold (hard requirement #2)."""
    cookies = creds.get_decrypted_cookies(db, user)
    if cookies is None:
        raise HTTPException(404, "No ESPN credentials connected.")
    s2, swid = cookies
    try:
        valid = validate_cookies(s2, swid)
    except Exception:  # noqa: BLE001 — transient; don't count as an auth failure
        raise HTTPException(502, "Couldn't reach ESPN. Please try again.")
    if valid:
        creds.record_auth_success(db, user)
        return {"ok": True, "valid": True, "leagues_found": _discover_safe(s2, swid)}
    purged = creds.record_auth_failure(db, user)
    return {"ok": True, "valid": False, "purged": purged}


@router.delete("/espn/credentials")
def espn_delete_credentials(user: User = Depends(get_current_user),
                            db: Session = Depends(get_db)) -> dict:
    """Delete only the ESPN cookies (immediate)."""
    removed = creds.delete_credentials(db, user)
    return {"ok": True, "connected": False, "removed": removed}


@router.delete("/account")
def delete_account(user: User = Depends(get_current_user),
                   db: Session = Depends(get_db)) -> dict:
    """Delete the whole account (immediate). Cascades to credentials, leagues,
    snapshots, proposals, and chat usage."""
    db.delete(user)
    db.commit()
    log.info("Deleted account %s", user.id)
    return {"ok": True, "deleted": True}
