"""LLM enrichment of news signals (Claude structured tool-use).

Already-sourced ESPN/Sleeper signals are enriched with *beneficiaries* (who gains
fantasy value from this news) using Claude with a forced-tool schema, so the
output is structured and validated. Bounded to a few high-severity signals per
cycle and gated on ANTHROPIC_API_KEY — without a key (or on any error) we fall
back to the deterministic signals unchanged. Claude never drives any write.
"""

from __future__ import annotations

import logging

from fantasy.config import settings
from fantasy.news.models import EventType, PlayerSignal

log = logging.getLogger(__name__)

# Cheap, fast model for extraction.
MODEL = "claude-haiku-4-5-20251001"

_TOOL = {
    "name": "report_impact",
    "description": "Report the fantasy-football impact of a news item.",
    "input_schema": {
        "type": "object",
        "properties": {
            "beneficiaries": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Player who gains value"},
                        "reason": {"type": "string"},
                    },
                    "required": ["name"],
                },
            },
            "fantasy_confidence": {"type": "number", "description": "0..1 confidence of impact"},
        },
        "required": ["beneficiaries"],
    },
}

# High-impact events worth an LLM call.
_ENRICH = {EventType.injury_out, EventType.ir, EventType.injury_doubtful,
           EventType.depth_change, EventType.trade}


def llm_available() -> bool:
    return bool(settings.anthropic_api_key)


def enrich_signals(signals: list[PlayerSignal], max_calls: int = 5) -> list[PlayerSignal]:
    if not llm_available():
        return signals
    try:
        from anthropic import Anthropic
    except ImportError:
        return signals
    client = Anthropic(api_key=settings.anthropic_api_key)

    calls = 0
    ranked = sorted(signals, key=lambda s: s.severity, reverse=True)
    for sig in ranked:
        if calls >= max_calls or sig.event_type not in _ENRICH:
            continue
        calls += 1
        try:
            prompt = (
                f"NFL news: {sig.player_name} ({sig.position or '?'}, {sig.team or '?'}). "
                f"Event: {sig.event_type.value}. Detail: {sig.summary}\n"
                "Which other fantasy-relevant players gain value as a direct result? "
                "Call report_impact with beneficiaries (only real, plausible players)."
            )
            resp = client.messages.create(
                model=MODEL, max_tokens=400, tools=[_TOOL],
                tool_choice={"type": "tool", "name": "report_impact"},
                messages=[{"role": "user", "content": prompt}],
            )
            for block in resp.content:
                if getattr(block, "type", None) == "tool_use":
                    data = block.input
                    sig.beneficiaries = [b["name"] for b in data.get("beneficiaries", []) if b.get("name")]
                    if "fantasy_confidence" in data:
                        sig.confidence = max(sig.confidence, float(data["fantasy_confidence"]))
        except Exception as e:  # noqa: BLE001
            log.warning("LLM enrich failed for %s: %s", sig.player_name, e)
    return signals
