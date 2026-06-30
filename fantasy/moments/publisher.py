"""Post an approved moment to the league group chat (Discord webhook).

A Discord channel webhook is the entire integration: one secret URL, no bot, no
OAuth. A single multipart POST sends the caption plus the PNG as a native
attachment (renders inline in the channel). ``?wait=true`` makes Discord return
the created message so we can record its id.

This runs from the approval hook (fantasy.api.app.on_approved) — i.e. only after
a human approves the moment. It never posts on its own.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import requests

from fantasy.config import settings
from fantasy.orchestrator.models import Proposal

log = logging.getLogger(__name__)

_MAX_CONTENT = 2000  # Discord message content hard limit


class DiscordPublisher:
    def __init__(self, webhook_url: str | None = None):
        self.webhook_url = webhook_url or settings.discord_webhook_url

    def publish(self, caption: str, image_path: str | None = None) -> str | None:
        if not self.webhook_url:
            log.warning("No DISCORD_WEBHOOK_URL configured — cannot post moment.")
            return None
        url = self.webhook_url + ("&" if "?" in self.webhook_url else "?") + "wait=true"
        data = {"payload_json": json.dumps({"content": (caption or "")[:_MAX_CONTENT]})}
        files = None
        if image_path and Path(image_path).exists():
            files = {"file": (Path(image_path).name, Path(image_path).read_bytes(), "image/png")}
        try:
            resp = requests.post(url, data=data, files=files, timeout=30)
            resp.raise_for_status()
            msg_id = (resp.json() or {}).get("id", "") if resp.content else ""
            return f"discord:{msg_id}"
        except Exception as e:  # noqa: BLE001
            log.warning("Discord publish failed: %s", e)
            return None


def publish_moment(proposal: Proposal) -> str | None:
    """Post a ``moment`` proposal's stored caption + graphic. Returns a message ref."""
    caption = proposal.payload.get("caption") or proposal.detail or proposal.title
    image_path = proposal.payload.get("image_path")
    return DiscordPublisher().publish(caption, image_path)
