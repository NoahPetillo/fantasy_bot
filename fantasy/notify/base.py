"""Notifier abstraction + text rendering shared by all channels."""

from __future__ import annotations

from typing import Protocol

from fantasy.orchestrator.models import Proposal

_ICON = {"start_sit": "🪑", "waiver": "➕", "trade": "🔁", "alert": "📰"}


def render_text(p: Proposal) -> str:
    icon = _ICON.get(p.kind.value, "•")
    star = "⭐ PRIORITY  " if p.payload.get("priority") else ""
    head = f"{star}{icon}  [{p.kind.value.upper()}] {p.title}"
    meta = f"value {p.value:+.1f} · confidence {p.confidence*100:.0f}% · wk {p.week}"
    return f"{head}\n{meta}\n{p.detail}"


def approve_reject_hint(p: Proposal) -> str:
    return f"Approve: ✅  ·  Reject: ❌   (id {p.id})"


class Notifier(Protocol):
    def notify(self, proposal: Proposal) -> str | None:
        """Send one proposal; return a channel-specific message ref (or None)."""
        ...


def get_notifier() -> Notifier:
    """Pick a channel from config: Slack > ntfy > console."""
    from fantasy.config import settings

    if settings.slack_bot_token and settings.slack_channel_id:
        from fantasy.notify.slack import SlackNotifier

        return SlackNotifier()
    if settings.ntfy_topic:
        from fantasy.notify.ntfy import NtfyNotifier

        return NtfyNotifier()
    from fantasy.notify.console import ConsoleNotifier

    return ConsoleNotifier()
