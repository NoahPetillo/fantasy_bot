"""Chatbot rate limit: anonymous callers are capped per IP; the authenticated
owner is exempt; limit=0 disables it. The LLM call is stubbed so these never hit
the network."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import fantasy.api.app as api
import fantasy.chat.agent as chat_agent
from fantasy.api import ratelimit
from fantasy.config import settings
from fantasy.orchestrator.store import Store


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "site_password", "letmein")
    api._store = Store(tmp_path / "rl.sqlite")
    # Stub answer() so /api/chat returns instantly with no LLM/network call.
    monkeypatch.setattr(chat_agent, "answer",
                        lambda q, ctx: {"answer": "ok", "tools_used": [], "mode": "test"})
    yield TestClient(api.app)
    api._chat_limiter = None  # don't leak a tiny limiter into other tests


def _ask(client):
    return client.post("/api/chat", json={"question": "hi"})


def test_anonymous_chat_is_capped(client):
    api._chat_limiter = ratelimit.RateLimiter(limit=3, window=3600)
    assert [_ask(client).status_code for _ in range(3)] == [200, 200, 200]
    r = _ask(client)
    assert r.status_code == 429
    assert r.headers.get("Retry-After")
    assert "try again" in r.json()["detail"].lower()


def test_authenticated_owner_is_exempt(client):
    api._chat_limiter = ratelimit.RateLimiter(limit=1, window=3600)
    client.post("/api/login", json={"password": "letmein"})  # become the owner
    assert [_ask(client).status_code for _ in range(5)] == [200] * 5


def test_limit_zero_disables(client):
    api._chat_limiter = ratelimit.RateLimiter(limit=0, window=3600)
    assert [_ask(client).status_code for _ in range(10)] == [200] * 10


def test_separate_ips_have_separate_budgets():
    rl = ratelimit.RateLimiter(limit=2, window=3600)
    assert rl.check("1.1.1.1")[0] and rl.check("1.1.1.1")[0]
    assert rl.check("1.1.1.1")[0] is False      # third from same IP blocked
    assert rl.check("2.2.2.2")[0] is True        # a different IP still has budget
