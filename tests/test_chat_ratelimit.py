"""Chat is now a logged-in feature; the per-IP limiter stays as an abuse floor
(a per-user plan quota lands in Phase 5). The LLM call is stubbed so these never
hit the network."""

from __future__ import annotations

import pytest

import fantasy.api.app as api
import fantasy.chat.agent as chat_agent
from fantasy.api import ratelimit


@pytest.fixture
def chat_client(webapp, monkeypatch):
    monkeypatch.setattr(chat_agent, "answer",
                        lambda q, ctx: {"answer": "ok", "tools_used": [], "mode": "test"})
    webapp.auth_as(webapp.make_user("chatter"))
    yield webapp.client
    api._chat_limiter = None


def _ask(client):
    return client.post("/api/chat", json={"question": "hi"})


def test_chat_is_capped_per_ip(chat_client):
    api._chat_limiter = ratelimit.RateLimiter(limit=3, window=3600)
    assert [_ask(chat_client).status_code for _ in range(3)] == [200, 200, 200]
    r = _ask(chat_client)
    assert r.status_code == 429
    assert r.headers.get("Retry-After")
    assert "try again" in r.json()["detail"].lower()


def test_limit_zero_disables(chat_client):
    api._chat_limiter = ratelimit.RateLimiter(limit=0, window=3600)
    assert [_ask(chat_client).status_code for _ in range(10)] == [200] * 10


def test_separate_ips_have_separate_budgets():
    rl = ratelimit.RateLimiter(limit=2, window=3600)
    assert rl.check("1.1.1.1")[0] and rl.check("1.1.1.1")[0]
    assert rl.check("1.1.1.1")[0] is False      # third from same IP blocked
    assert rl.check("2.2.2.2")[0] is True        # a different IP still has budget
