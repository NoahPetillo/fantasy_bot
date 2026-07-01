"""Shared-password gate: protected endpoints 401 when locked; chatbot + login
stay open; a correct password unlocks; no password means the gate is off."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import fantasy.api.app as api
from fantasy.api import auth
from fantasy.config import settings
from fantasy.orchestrator.store import Store


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "site_password", "letmein")
    api._store = Store(tmp_path / "gate.sqlite")  # fresh store, never the real one
    return TestClient(api.app)  # TestClient persists cookies across requests


def test_protected_endpoints_blocked_when_locked(client):
    assert client.get("/api/dashboard").status_code == 401
    assert client.get("/api/proposals").status_code == 401
    assert client.get("/api/leagues").status_code == 401


def test_public_paths_stay_open_when_locked(client):
    assert client.get("/health").status_code == 200
    assert client.get("/").status_code == 200          # the HTML shell
    assert client.get("/api/session").status_code == 200
    assert auth.is_public("/api/chat")                  # the one feature league mates get


def test_wrong_password_is_rejected(client):
    assert client.post("/api/login", json={"password": "nope"}).status_code == 401
    assert client.get("/api/proposals").status_code == 401  # still locked


def test_correct_password_unlocks_then_logout_relocks(client):
    assert client.get("/api/session").json()["authed"] is False
    r = client.post("/api/login", json={"password": "letmein"})
    assert r.status_code == 200 and r.json()["ok"] is True
    assert auth.COOKIE in client.cookies
    assert client.get("/api/proposals").status_code == 200      # cookie lets us through
    assert client.get("/api/session").json()["authed"] is True
    client.post("/api/logout")
    assert client.get("/api/proposals").status_code == 401      # re-locked


def test_gate_off_when_no_password(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "site_password", None)
    api._store = Store(tmp_path / "open.sqlite")
    c = TestClient(api.app)
    assert c.get("/api/proposals").status_code == 200
    assert c.get("/api/session").json() == {"gate_enabled": False, "authed": True}
