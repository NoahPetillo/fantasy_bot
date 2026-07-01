"""Shared-password gate for the web surface.

A single site-wide password (``SITE_PASSWORD`` in the environment) protects every
endpoint except the public ones: the chatbot (``/api/chat``), the login flow,
``/health``, and the static shell. League mates get the chatbot; the rest of the
dashboard is private.

Auth is a **stateless HMAC-signed cookie** — there is no server-side session
store. The cookie is ``base64(exp . hmac_sha256(SITE_PASSWORD, exp))``; we trust
it because only someone who knows the password could have minted a valid
signature. This means it survives restarts, needs no database (the existing
SQLite store stays untouched), and works on serverless where nothing persists
between requests. Rotating ``SITE_PASSWORD`` changes the signing key, so every
previously-issued cookie stops validating — an instant "log everyone out".

When ``SITE_PASSWORD`` is unset the gate is OFF and the app behaves exactly as it
did before (handy for local dev). Settings are read at call time, so the gate can
be toggled per-process via the environment without code changes.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import time

from fantasy.config import settings

COOKIE = "fa_session"
TTL = 30 * 86400  # cookie lifetime: 30 days

# Reachable WITHOUT the password. Everything else is gated.
#   /                  the HTML shell (just markup + JS; carries no league data)
#   /api/chat          the one feature league mates are meant to use
#   /api/session       lets the frontend decide whether to show the lock screen
#   /api/login|logout  the gate itself
#   /slack/interactions Slack's servers can't send our cookie; that channel is
#                       authenticated separately by Slack's signing secret.
_PUBLIC_EXACT = {
    "/", "/health", "/api/chat", "/api/login", "/api/logout", "/api/session",
    "/slack/interactions", "/favicon.ico", "/openapi.json", "/docs", "/redoc",
}
_PUBLIC_PREFIX = ("/static/",)


def gate_enabled() -> bool:
    """True when a password is configured (and the gate should enforce it)."""
    return bool(settings.site_password)


def is_public(path: str) -> bool:
    return path in _PUBLIC_EXACT or path.startswith(_PUBLIC_PREFIX)


def check_password(pw: str) -> bool:
    """Constant-time compare of a submitted password against the configured one."""
    sp = settings.site_password or ""
    return bool(sp) and hmac.compare_digest(pw.encode(), sp.encode())


def _secret() -> bytes:
    return (settings.site_password or "").encode()


def issue_token(ttl: int = TTL) -> str:
    """Mint a signed session token good for ``ttl`` seconds."""
    exp = str(int(time.time()) + ttl)
    sig = hmac.new(_secret(), exp.encode(), hashlib.sha256).hexdigest()
    return base64.urlsafe_b64encode(f"{exp}.{sig}".encode()).decode()


def valid_token(token: str | None) -> bool:
    """True iff ``token`` is a well-formed, unexpired, correctly-signed cookie."""
    if not token:
        return False
    try:
        raw = base64.urlsafe_b64decode(token.encode()).decode()
        exp_s, sig = raw.split(".", 1)
        exp = int(exp_s)
    except (ValueError, TypeError):
        return False
    if exp < int(time.time()):
        return False
    good = hmac.new(_secret(), exp_s.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(good, sig)
