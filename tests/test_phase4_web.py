"""Phase 4 — frontend/backend glue: /api/config, legal pages, and confirmation
that the shared-password gate is gone (endpoints are Clerk-only)."""

from __future__ import annotations

import base64

from fastapi.testclient import TestClient

import fantasy.api.app as api
from fantasy.api import clerk_auth
from fantasy.config import settings


def _client():
    return TestClient(api.app)


def test_config_reports_unconfigured_by_default():
    body = _client().get("/api/config").json()
    assert body["auth_configured"] is False
    assert body["clerk_publishable_key"] is None
    assert body["clerk_frontend_api"] is None


def test_config_derives_frontend_api_from_publishable_key(monkeypatch):
    host = "relaxed-cat-42.clerk.accounts.dev"
    pk = "pk_test_" + base64.b64encode(f"{host}$".encode()).decode()
    monkeypatch.setattr(settings, "clerk_publishable_key", pk)
    body = _client().get("/api/config").json()
    assert body["clerk_frontend_api"] == host
    assert body["auth_configured"] is True
    assert clerk_auth.clerk_issuer() == f"https://{host}"


def test_privacy_page_renders_with_disclaimer():
    r = _client().get("/privacy")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    html = r.text
    assert "Privacy Policy" in html
    assert settings.product_name in html and "[PRODUCT_NAME]" not in html
    assert "not affiliated" in html.lower()


def test_terms_page_renders():
    r = _client().get("/terms")
    assert r.status_code == 200 and "Terms of Service" in r.text
    assert "not affiliated" in r.text.lower()


def test_password_gate_is_gone():
    c = _client()
    # No login/session endpoints anymore.
    assert c.post("/api/login", json={"password": "x"}).status_code in (404, 405)
    assert c.get("/api/session").status_code == 404
    # Protected endpoints now fail with the Clerk 401 (never "password required").
    r = c.get("/api/leagues")
    assert r.status_code == 401 and r.json()["detail"] == "authentication required"


def test_public_pages_open_without_auth():
    c = _client()
    for path in ("/health", "/", "/connect", "/privacy", "/terms", "/api/config"):
        assert c.get(path).status_code == 200, path
