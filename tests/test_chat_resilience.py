"""Chat must never 500 and should still answer snapshot-based questions when the
LLM is rate-limited AND the stats source is unavailable (e.g. offseason: this
season's nflverse data doesn't exist yet). Reproduces the Groq-429 + 404 case."""

from __future__ import annotations

import fantasy.chat.agent as agent
from fantasy.chat import tools as chat_tools
from fantasy.chat.tools import ChatContext
from fantasy.config import settings

SNAP = {
    "team": {"season": 2026},
    "league_settings": {"summary": "12-team PPR",
                        "scoring": {"receptions": 1.0, "passing_yards": 0.04},
                        "roster": {}},
}


def _stats_unavailable(monkeypatch):
    def boom(seasons):
        raise ConnectionError("404 Not Found (season data not published yet)")
    monkeypatch.setattr(chat_tools, "load_weekly", boom)


def test_weekly_degrades_to_empty_when_unavailable(monkeypatch):
    _stats_unavailable(monkeypatch)
    df = ChatContext.from_snapshot(SNAP).weekly()
    assert df.empty  # no crash


def test_find_player_tolerates_empty_frame():
    import pandas as pd
    assert chat_tools.find_player(pd.DataFrame(), "some player") == (None, None)


def test_scoring_rules_answered_without_llm_or_stats(monkeypatch):
    # No LLM keys → keyless parser; stats source down → offseason.
    monkeypatch.setattr(settings, "groq_api_key", None)
    monkeypatch.setattr(settings, "anthropic_api_key", None)
    _stats_unavailable(monkeypatch)
    out = agent.answer("What are the league scoring rules?", ChatContext.from_snapshot(SNAP))
    assert out["mode"] != "error"                      # it actually answered
    assert "receptions" in out["answer"].lower() or "ppr" in out["answer"].lower()


def test_answer_never_raises_even_if_fallback_fails(monkeypatch):
    monkeypatch.setattr(settings, "groq_api_key", None)
    monkeypatch.setattr(settings, "anthropic_api_key", None)
    monkeypatch.setattr(agent, "_fallback", lambda q, ctx: (_ for _ in ()).throw(RuntimeError("boom")))
    out = agent.answer("anything", ChatContext.from_snapshot(SNAP))
    assert out["mode"] == "error" and "try again" in out["answer"].lower()


def test_answer_falls_back_to_keyless_when_groq_errors(monkeypatch):
    monkeypatch.setattr(settings, "groq_api_key", "gsk_fake")
    monkeypatch.setattr(settings, "anthropic_api_key", None)
    _stats_unavailable(monkeypatch)

    def groq_429(question, ctx):
        raise RuntimeError("Error code: 429 - rate_limit_exceeded")
    monkeypatch.setattr(agent, "_groq", groq_429)
    out = agent.answer("What are the league scoring rules?", ChatContext.from_snapshot(SNAP))
    assert out["mode"].startswith("fallback")          # degraded, not 500
    assert out["answer"]
