"""Legacy shared-password gate (removed in Phase 4). It still blocks non-public
paths when locked; per-user endpoints ALSO require Clerk, so once the gate passes
the Clerk layer takes over (401 "authentication required")."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import fantasy.api.app as api
from fantasy.api import auth
from fantasy.config import settings


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "site_password", "letmein")
    return TestClient(api.app)  # persists cookies across requests


def test_gate_blocks_non_public_when_locked(client):
    r = client.get("/api/leagues")
    assert r.status_code == 401 and r.json()["detail"] == "password required"


def test_public_paths_stay_open_when_locked(client):
    assert client.get("/health").status_code == 200
    assert client.get("/").status_code == 200                 # HTML shell
    assert client.get("/api/session").status_code == 200
    assert client.get("/connect").status_code == 200          # connect shell
    assert client.get("/api/legal/espn-consent").status_code == 200
    assert auth.is_public("/api/chat")


def test_wrong_password_is_rejected(client):
    assert client.post("/api/login", json={"password": "nope"}).status_code == 401
    assert client.get("/api/leagues").json()["detail"] == "password required"  # still locked


def test_correct_password_passes_gate_then_clerk_takes_over(client):
    assert client.get("/api/session").json()["authed"] is False
    r = client.post("/api/login", json={"password": "letmein"})
    assert r.status_code == 200 and auth.COOKIE in client.cookies
    assert client.get("/api/session").json()["authed"] is True
    # Gate now passes; the endpoint requires Clerk, so it's the Clerk layer that 401s.
    after = client.get("/api/leagues")
    assert after.status_code == 401 and after.json()["detail"] == "authentication required"


def test_gate_off_when_no_password(monkeypatch):
    monkeypatch.setattr(settings, "site_password", None)
    c = TestClient(api.app)
    assert c.get("/api/session").json() == {"gate_enabled": False, "authed": True}
    # Gate off, but the endpoint still requires Clerk auth.
    assert c.get("/api/leagues").json()["detail"] == "authentication required"
