"""Clerk-verified ``current_user`` dependency (managed auth — we do NOT hand-roll).

The frontend authenticates with Clerk and sends the short-lived Clerk **session
JWT** as ``Authorization: Bearer <jwt>`` (or Clerk's ``__session`` cookie). Here we
verify that JWT against Clerk's JWKS (RS256, per-request, keys cached), then map
the token's ``sub`` (the Clerk user id) to a row in our ``users`` table — creating
it on first login. Every protected route depends on :func:`get_current_user`, so
each request is scoped to exactly one authenticated user (hard requirement #1).

This coexists with the legacy shared-password gate during the migration; the
password gate is removed in Phase 4.
"""

from __future__ import annotations

import logging

import jwt
from fastapi import Depends, HTTPException, Request
from jwt import PyJWKClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from fantasy.config import settings
from fantasy.db.base import get_db
from fantasy.db.models import User

log = logging.getLogger(__name__)

_CLOCK_LEEWAY = 10  # seconds, tolerate minor clock skew
_jwk_clients: dict[str, PyJWKClient] = {}


def bearer_token(request: Request) -> str | None:
    """Pull the Clerk session JWT from the Authorization header or ``__session``."""
    header = request.headers.get("authorization") or request.headers.get("Authorization")
    if header and header.lower().startswith("bearer "):
        return header[7:].strip() or None
    return request.cookies.get("__session")


def _jwks_url() -> str:
    """The JWKS endpoint — from TRUSTED CONFIG ONLY, never from the token.

    Deriving the JWKS URL from the token's own ``iss`` would let an attacker point
    us at *their* JWKS and sign their own tokens (issuer spoofing → full auth
    bypass). We fail closed if neither ``CLERK_JWKS_URL`` nor ``CLERK_ISSUER`` is
    configured.
    """
    if settings.clerk_jwks_url:
        return settings.clerk_jwks_url
    if settings.clerk_issuer:
        return settings.clerk_issuer.rstrip("/") + "/.well-known/jwks.json"
    raise HTTPException(503, "Auth is not configured (set CLERK_ISSUER or CLERK_JWKS_URL).")


def _signing_key(token: str):
    """Fetch the RSA signing key for ``token`` from the configured JWKS (cached).
    The URL comes from config, so keys are only ever trusted from Clerk."""
    url = _jwks_url()
    client = _jwk_clients.get(url)
    if client is None:
        client = PyJWKClient(url)
        _jwk_clients[url] = client
    return client.get_signing_key_from_jwt(token).key


def verify_clerk_token(token: str) -> dict:
    """Verify a Clerk session JWT and return its claims. Raises 401 on any failure.

    Signature is checked against keys from the *configured* JWKS (never the
    token's own issuer). When ``CLERK_ISSUER`` is set, the ``iss`` claim must be
    present and match it.
    """
    require = ["exp", "sub"]
    decode_kwargs: dict = {}
    if settings.clerk_issuer:
        decode_kwargs["issuer"] = settings.clerk_issuer
        require.append("iss")  # must be present AND equal the configured issuer
    try:
        key = _signing_key(token)
        claims = jwt.decode(
            token,
            key,
            algorithms=["RS256"],
            leeway=_CLOCK_LEEWAY,
            options={"verify_aud": False, "require": require},
            **decode_kwargs,
        )
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001 — any verification failure is a 401
        log.info("Clerk token rejected: %s", type(e).__name__)
        raise HTTPException(401, "invalid or expired session")
    if not claims.get("sub"):
        raise HTTPException(401, "session token missing subject")
    return claims


def get_or_create_user(db: Session, clerk_user_id: str, email: str | None = None) -> User:
    """Upsert on first login: return the user for ``clerk_user_id``, creating it if
    absent. Backfills a newly-known email onto an existing row."""
    user = db.execute(
        select(User).where(User.clerk_user_id == clerk_user_id)
    ).scalar_one_or_none()
    if user is None:
        user = User(clerk_user_id=clerk_user_id, email=email)
        db.add(user)
        db.commit()
        db.refresh(user)
        log.info("Provisioned new user for Clerk id %s", clerk_user_id)
        return user
    if email and not user.email:
        user.email = email
        db.commit()
        db.refresh(user)
    return user


def get_current_user(request: Request, db: Session = Depends(get_db)) -> User:
    """FastAPI dependency: the authenticated :class:`User` for this request.

    401 if no/invalid Clerk token. Tests override this via
    ``app.dependency_overrides[get_current_user]``.
    """
    token = bearer_token(request)
    if not token:
        raise HTTPException(401, "authentication required")
    claims = verify_clerk_token(token)
    return get_or_create_user(db, claims["sub"], claims.get("email"))
