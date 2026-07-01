"""Clerk-verified current_user dependency + users-upsert-on-first-login."""

from __future__ import annotations

import time

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient
from sqlalchemy import select

import fantasy.api.app as api
from fantasy.api import clerk_auth
from fantasy.config import settings
from fantasy.db.models import User


@pytest.fixture(scope="module")
def rsa_keys():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    priv_pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    return priv_pem, key.public_key()


def _make_token(priv_pem, *, sub="clerk_user_123", email=None, exp_delta=3600,
                iss="https://clerk.test"):
    now = int(time.time())
    claims = {"sub": sub, "iat": now, "nbf": now, "exp": now + exp_delta, "iss": iss}
    if email:
        claims["email"] = email
    return jwt.encode(claims, priv_pem, algorithm="RS256")


# ── unit: get_or_create_user (the "users upsert on first login" requirement) ──
def test_get_or_create_user_provisions_then_is_idempotent(db):
    u1 = clerk_auth.get_or_create_user(db, "clerk_abc", "a@ex.com")
    assert u1.id is not None and u1.plan == "free"
    u2 = clerk_auth.get_or_create_user(db, "clerk_abc", "a@ex.com")
    assert u2.id == u1.id  # no duplicate row on second login
    assert db.execute(select(User).where(User.clerk_user_id == "clerk_abc")).scalars().all() == [u1]


def test_get_or_create_user_backfills_email(db):
    u = clerk_auth.get_or_create_user(db, "clerk_noemail", None)
    assert u.email is None
    u2 = clerk_auth.get_or_create_user(db, "clerk_noemail", "later@ex.com")
    assert u2.id == u.id and u2.email == "later@ex.com"


# ── verify_clerk_token ──
def test_verify_valid_token(rsa_keys, monkeypatch):
    priv, pub = rsa_keys
    monkeypatch.setattr(clerk_auth, "_signing_key", lambda token: pub)
    claims = clerk_auth.verify_clerk_token(_make_token(priv, sub="s1", email="e@ex.com"))
    assert claims["sub"] == "s1" and claims["email"] == "e@ex.com"


def test_verify_expired_token_401(rsa_keys, monkeypatch):
    priv, pub = rsa_keys
    monkeypatch.setattr(clerk_auth, "_signing_key", lambda token: pub)
    with pytest.raises(Exception) as ei:
        clerk_auth.verify_clerk_token(_make_token(priv, exp_delta=-3600))
    assert getattr(ei.value, "status_code", None) == 401


def test_verify_wrong_key_401(rsa_keys, monkeypatch):
    priv, _ = rsa_keys
    other_pub = rsa.generate_private_key(public_exponent=65537, key_size=2048).public_key()
    monkeypatch.setattr(clerk_auth, "_signing_key", lambda token: other_pub)  # signature won't match
    with pytest.raises(Exception) as ei:
        clerk_auth.verify_clerk_token(_make_token(priv))
    assert getattr(ei.value, "status_code", None) == 401


# ── issuer enforcement (fixes the critical issuer-spoofing findings) ──
def test_verify_rejects_issuer_mismatch(rsa_keys, monkeypatch):
    priv, pub = rsa_keys
    monkeypatch.setattr(settings, "clerk_issuer", "https://real.clerk.test")
    monkeypatch.setattr(clerk_auth, "_signing_key", lambda token: pub)
    # Correctly-signed token, but from a different (attacker) issuer.
    with pytest.raises(Exception) as ei:
        clerk_auth.verify_clerk_token(_make_token(priv, iss="https://attacker.example"))
    assert getattr(ei.value, "status_code", None) == 401


def test_verify_accepts_matching_issuer(rsa_keys, monkeypatch):
    priv, pub = rsa_keys
    monkeypatch.setattr(settings, "clerk_issuer", "https://real.clerk.test")
    monkeypatch.setattr(clerk_auth, "_signing_key", lambda token: pub)
    claims = clerk_auth.verify_clerk_token(_make_token(priv, iss="https://real.clerk.test"))
    assert claims["iss"] == "https://real.clerk.test"


def test_jwks_url_comes_from_config_never_from_token(rsa_keys, monkeypatch):
    """The JWKS endpoint must come from trusted config, NOT the token's iss —
    otherwise an attacker points us at their own JWKS and forges tokens."""
    priv, pub = rsa_keys
    monkeypatch.setattr(settings, "clerk_jwks_url", "https://trusted.example/jwks.json")
    monkeypatch.setattr(settings, "clerk_issuer", None)

    class _FakeJWKClient:
        last_url = None

        def __init__(self, url):
            _FakeJWKClient.last_url = url

        def get_signing_key_from_jwt(self, token):
            return type("K", (), {"key": pub})()

    monkeypatch.setattr(clerk_auth, "PyJWKClient", _FakeJWKClient)
    clerk_auth._jwk_clients.clear()

    # Token claims a hostile issuer; the app must ignore it for key lookup.
    clerk_auth._signing_key(_make_token(priv, iss="https://attacker.example"))
    assert _FakeJWKClient.last_url == "https://trusted.example/jwks.json"


def test_auth_fails_closed_when_unconfigured(monkeypatch):
    monkeypatch.setattr(settings, "clerk_issuer", None)
    monkeypatch.setattr(settings, "clerk_jwks_url", None)
    with pytest.raises(Exception) as ei:
        clerk_auth._jwks_url()
    assert getattr(ei.value, "status_code", None) == 503


# ── integration: /api/me end-to-end (token → verify → upsert → row) ──
def test_api_me_requires_token(db):
    client = TestClient(api.app)
    assert client.get("/api/me").status_code == 401


def test_api_me_provisions_and_returns_user(db, rsa_keys, monkeypatch):
    priv, pub = rsa_keys
    monkeypatch.setattr(clerk_auth, "_signing_key", lambda token: pub)
    token = _make_token(priv, sub="clerk_me", email="me@ex.com")
    client = TestClient(api.app)

    r = client.get("/api/me", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["clerk_user_id"] == "clerk_me" and body["email"] == "me@ex.com" and body["plan"] == "free"

    # Second call with the same subject must not create a second user row.
    client.get("/api/me", headers={"Authorization": f"Bearer {token}"})
    rows = db.execute(select(User).where(User.clerk_user_id == "clerk_me")).scalars().all()
    assert len(rows) == 1


def test_api_me_accepts_session_cookie(db, rsa_keys, monkeypatch):
    priv, pub = rsa_keys
    monkeypatch.setattr(clerk_auth, "_signing_key", lambda token: pub)
    token = _make_token(priv, sub="clerk_cookie")
    client = TestClient(api.app)
    r = client.get("/api/me", cookies={"__session": token})
    assert r.status_code == 200 and r.json()["clerk_user_id"] == "clerk_cookie"
