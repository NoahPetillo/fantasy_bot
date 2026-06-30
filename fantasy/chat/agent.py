"""The chatbot agent — routes a question to the data tools and phrases the answer.

Provider preference: Groq (fast open models, OpenAI-style tool calling) > Anthropic
(Claude tool use) > a deterministic keyless parser. In every case the model only
SELECTS tools and writes the sentence — the numbers come from the tools, so answers
stay grounded regardless of provider.
"""

from __future__ import annotations

import json
import logging
import re

from fantasy.chat.tools import STAT_ALIASES, TOOLS, ChatContext, find_player, run_tool
from fantasy.config import settings

log = logging.getLogger(__name__)

# Haiku: the established model for this repo's bounded-LLM features, and the right
# tier for a tool-router that just selects tools and writes one sentence.
MODEL = "claude-haiku-4-5-20251001"
MAX_TURNS = 6

SYSTEM = (
    "You are a concise fantasy-football analyst for the user's league. Answer NFL "
    "stat and league questions using ONLY the provided tools — never state a number "
    "you didn't get from a tool. The current season is {season}. For a 'since <event>' "
    "question (e.g. since a player got injured), first call get_player_absences to find "
    "the event week, then aggregate from that week with get_player_stat. Give the number "
    "first, then a short clause of context. For comparisons, call the tool for EACH "
    "player and state both numbers. If a player can't be found, say so plainly."
)


def answer(question: str, ctx: ChatContext) -> dict:
    """Return {answer, mode, tools_used}. Never raises — degrades to a helpful message."""
    if not question or not question.strip():
        return {"answer": "Ask me an NFL or league question.", "mode": "noop", "tools_used": []}
    provider = ("_groq" if settings.groq_api_key else
                "_llm" if settings.anthropic_api_key else None)
    if provider is None:
        return _fallback(question, ctx)
    try:
        return (_groq if provider == "_groq" else _llm)(question, ctx)
    except Exception as e:  # noqa: BLE001
        log.warning("chat LLM (%s) failed (%s); using fallback.", provider, e)
        out = _fallback(question, ctx)
        out["mode"] = "fallback (LLM error)"
        return out


def _openai_tools() -> list[dict]:
    """Our Anthropic-style tool schemas as OpenAI/Groq function definitions."""
    return [{"type": "function",
             "function": {"name": t["name"], "description": t["description"],
                          "parameters": t["input_schema"]}} for t in TOOLS]


def _groq(question: str, ctx: ChatContext) -> dict:
    from groq import Groq

    client = Groq(api_key=settings.groq_api_key)
    messages = [{"role": "system", "content": SYSTEM.format(season=ctx.season)},
                {"role": "user", "content": question}]
    used: list[str] = []
    final = ""
    for _ in range(MAX_TURNS):
        resp = client.chat.completions.create(
            model=settings.groq_model, messages=messages, tools=_openai_tools(),
            tool_choice="auto", temperature=0, max_tokens=700)
        msg = resp.choices[0].message
        final = (msg.content or "").strip()
        calls = msg.tool_calls or []
        if not calls:
            break
        messages.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [{"id": c.id, "type": "function",
                                         "function": {"name": c.function.name,
                                                      "arguments": c.function.arguments}}
                                        for c in calls]})
        for c in calls:
            used.append(c.function.name)
            try:
                args = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            messages.append({"role": "tool", "tool_call_id": c.id,
                             "content": run_tool(c.function.name, args, ctx)})
    return {"answer": final or "I couldn't find an answer to that.",
            "mode": "groq", "tools_used": used}


def _llm(question: str, ctx: ChatContext) -> dict:
    from anthropic import Anthropic

    client = Anthropic(api_key=settings.anthropic_api_key)
    system = SYSTEM.format(season=ctx.season)
    messages = [{"role": "user", "content": question}]
    used: list[str] = []
    final = ""
    for _ in range(MAX_TURNS):
        resp = client.messages.create(model=MODEL, max_tokens=700, system=system,
                                      tools=TOOLS, messages=messages)
        final = " ".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()
        if resp.stop_reason != "tool_use":
            break
        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for b in resp.content:
            if getattr(b, "type", None) == "tool_use":
                used.append(b.name)
                results.append({"type": "tool_result", "tool_use_id": b.id,
                                "content": run_tool(b.name, b.input, ctx)})
        messages.append({"role": "user", "content": results})
    return {"answer": final or "I couldn't find an answer to that.",
            "mode": "llm", "tools_used": used}


# ── keyless fallback ──────────────────────────────────────────────────────────
def _find_player(text: str, ctx: ChatContext) -> str | None:
    gid, disp = find_player(ctx.weekly(), text or "")
    return disp if gid else None


def _fallback(question: str, ctx: ChatContext) -> dict:
    q = question.lower()
    # Split on since/after/from: the SUBJECT player is on the left, an event
    # reference (e.g. "since <other> got injured") is on the right.
    parts = re.split(r"\b(?:since|after|from)\b", question, maxsplit=1, flags=re.I)
    subject_text, event_text = parts[0], (parts[1] if len(parts) > 1 else "")
    player = _find_player(subject_text, ctx) or _find_player(question, ctx)
    tools_used: list[str] = []

    if "project" in q and player:
        tools_used.append("get_projection")
        return {"answer": run_tool("get_projection", {"player": player}, ctx),
                "mode": "fallback", "tools_used": tools_used}
    if any(w in q for w in ("scoring", "settings", "my league", "league score", "roster slots", "ppr")) and not player:
        return {"answer": run_tool("get_league_settings", {}, ctx),
                "mode": "fallback", "tools_used": ["get_league_settings"]}

    if player:
        stat = _detect_stat(q)
        frm, to = _window(question, event_text, ctx, tools_used)
        if stat:
            tools_used.append("get_player_stat")
            payload = {"player": player, "stat": stat, "from_week": frm}
            if to is not None:
                payload["to_week"] = to
            return {"answer": run_tool("get_player_stat", payload, ctx),
                    "mode": "fallback", "tools_used": tools_used}
        tools_used.append("get_player_game_log")
        return {"answer": run_tool("get_player_game_log", {"player": player}, ctx),
                "mode": "fallback", "tools_used": tools_used}

    return {"answer": "I can answer questions like 'how many TDs has <player> scored "
                      "since week 5?', '<player> in week 4', or 'how many points is "
                      "<player> projected?'. (Set ANTHROPIC_API_KEY for full free-form chat.)",
            "mode": "fallback", "tools_used": tools_used}


def _detect_stat(q: str) -> str | None:
    for key in sorted(STAT_ALIASES, key=len, reverse=True):  # multi-word keys first
        if key.replace("_", " ") in q or key in q:
            return key
    if "fantasy point" in q or "fantasy pts" in q or " points" in q or "how many points" in q:
        return "fantasy_points"
    if "yard" in q:
        return "receiving_yards" if "rec" in q else ("rushing_yards" if "rush" in q else "passing_yards")
    return None


def _window(question: str, event_text: str, ctx: ChatContext, tools_used: list):
    """(from_week, to_week|None) for a stat question's time range."""
    ql = question.lower()
    m = re.search(r"weeks?\s+(\d+)\s*(?:to|through|thru|and|-|–)\s*(?:week\s*)?(\d+)", ql)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r"(?:since|after|from)\s+week\s+(\d+)", ql)
    if m:
        return int(m.group(1)), None
    if event_text and ("injur" in event_text.lower() or "hurt" in event_text.lower()):
        w = _injury_week(event_text, ctx, tools_used)
        if w:
            return w, None
    m = re.search(r"\bweek\s+(\d+)", ql)  # a single specific week ("in week 4")
    if m and not re.search(r"(?:since|after|from)\s+week", ql):
        w = int(m.group(1))
        return w, w
    return 1, None


def _injury_week(event_text: str, ctx: ChatContext, tools_used: list) -> int | None:
    df = ctx.weekly()
    gid, _ = find_player(df, event_text)
    if not gid:
        return None
    played = sorted(int(w) for w in df.loc[df["player_id"] == gid, "week"].unique())
    if not played:
        return None
    last = min(int(df["week"].max()), 18)
    missed = [w for w in range(played[0], last + 1) if w not in set(played)]
    if missed:
        tools_used.append("get_player_absences")
        return missed[0]
    return None
