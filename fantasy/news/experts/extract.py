"""Extract structured ExpertSignals from raw posts.

Two paths, same output:
- LLM (Claude forced-tool, when ANTHROPIC_API_KEY is set): robust, flags
  sarcasm/hedge/hypothetical, infers the event + direction.
- Deterministic keyword fallback (no key): finds player names and classifies by
  keywords — enough to run + test the pipeline offline.

Both ground the player to a gsis id via the crosswalk and DROP names that won't
resolve (kills hallucinated / non-NFL names).
"""

from __future__ import annotations

import functools
import logging
import re

from fantasy.config import settings
from fantasy.data.ids import crosswalk, norm_name
from fantasy.news.experts.models import ExpertSignal, RawPost
from fantasy.news.models import EventType

log = logging.getLogger(__name__)

_NAME_RE = re.compile(r"\b([A-Z][a-zA-Z'’.\-]+(?:\.?\s[A-Z][a-zA-Z'’.\-]+){1,2})\b")

_KEYWORDS = [
    (("ruled out", "won't play", "wont play", "will not play", "inactive", "out for", "out vs"),
     EventType.injury_out, -1),
    (("placed on ir", "injured reserve", "season-ending"), EventType.ir, -1),
    (("doubtful",), EventType.injury_doubtful, -1),
    (("questionable",), EventType.injury_questionable, -1),
    (("returns", "will play", "active for", "cleared", "back at practice", "good to go"),
     EventType.injury_return, +1),
    (("lead back", "starting job", "promoted", "first-team", "every-down", "workhorse",
      "snap share", "route share", "target share", "more touches"), EventType.usage_change, +1),
    (("league winner", "must-add", "must add", "waiver gem", "pick him up", "add him",
      "priority add", "breakout"), EventType.breakout, +1),
    (("buy low", "buy-low"), EventType.buy_low, +1),
    (("sell high", "sell-high"), EventType.sell_high, -1),
]


@functools.lru_cache(maxsize=1)
def _name_index() -> dict[str, str]:
    xw = crosswalk()
    idx: dict[str, str] = {}
    for gid, nm in xw.gsis_to_name.items():
        if isinstance(nm, str) and " " in nm:
            idx[norm_name(nm)] = gid
    return idx


def _resolve(name: str) -> str | None:
    return _name_index().get(norm_name(name))


def _classify(text: str) -> tuple[EventType, int]:
    t = text.lower()
    for words, etype, direction in _KEYWORDS:
        if any(w in t for w in words):
            return etype, direction
    return EventType.news, 0


def extract_signals(posts: list[RawPost]) -> list[ExpertSignal]:
    if settings.anthropic_api_key:
        out: list[ExpertSignal] = []
        for p in posts:
            out.extend(_llm_extract(p) or _deterministic(p))
        return out
    return [s for p in posts for s in _deterministic(p)]


def _deterministic(post: RawPost) -> list[ExpertSignal]:
    etype, direction = _classify(post.text)
    if etype == EventType.news:
        return []
    signals = []
    seen = set()
    for m in _NAME_RE.finditer(post.text):
        gid = _resolve(m.group(1))
        if gid and gid not in seen:
            seen.add(gid)
            signals.append(ExpertSignal(
                player_name=m.group(1), player_id=gid, event_type=etype, direction=direction,
                confidence=0.6, expert_handle=post.author_handle, outlet=post.outlet,
                base_trust=post.base_trust, ts=post.ts, source_url=post.url,
            ))
    return signals


_TOOL = {
    "name": "report_signal",
    "description": "Report fantasy-relevant player signals in a post.",
    "input_schema": {
        "type": "object",
        "properties": {
            "signals": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "player_name": {"type": "string"},
                    "event_type": {"type": "string", "enum": [e.value for e in EventType]},
                    "direction": {"type": "integer", "enum": [-1, 0, 1]},
                    "confidence": {"type": "number"},
                    "is_sarcasm": {"type": "boolean"},
                    "is_hedge": {"type": "boolean"},
                    "is_hypothetical": {"type": "boolean"},
                },
                "required": ["player_name", "event_type", "direction"],
            }},
        },
        "required": ["signals"],
    },
}


def _llm_extract(post: RawPost) -> list[ExpertSignal] | None:
    try:
        from anthropic import Anthropic

        client = Anthropic(api_key=settings.anthropic_api_key)
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001", max_tokens=500, tools=[_TOOL],
            tool_choice={"type": "tool", "name": "report_signal"},
            messages=[{"role": "user", "content": (
                "Extract fantasy-football player signals from this post. Flag sarcasm, "
                "hedges, and hypotheticals. Only real NFL players.\n\nPOST: " + post.text)}],
        )
        out = []
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use":
                for s in block.input.get("signals", []):
                    gid = _resolve(s["player_name"])
                    if not gid:
                        continue
                    out.append(ExpertSignal(
                        player_name=s["player_name"], player_id=gid,
                        event_type=EventType(s["event_type"]), direction=int(s.get("direction", 0)),
                        confidence=float(s.get("confidence", 0.5)),
                        is_sarcasm=bool(s.get("is_sarcasm")), is_hedge=bool(s.get("is_hedge")),
                        is_hypothetical=bool(s.get("is_hypothetical")),
                        expert_handle=post.author_handle, outlet=post.outlet,
                        base_trust=post.base_trust, ts=post.ts, source_url=post.url,
                    ))
        return out
    except Exception as e:  # noqa: BLE001
        log.warning("LLM extract failed (%s); falling back.", e)
        return None
