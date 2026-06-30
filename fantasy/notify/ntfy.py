"""ntfy.sh notifier — zero-setup push with approve/reject action buttons.

The action buttons POST to the FastAPI approve/reject endpoints (set
``PUBLIC_BASE_URL`` if exposing them); without a public URL the notification is
still delivered and the user can approve from the dashboard.
"""

from __future__ import annotations

import logging
import os

import requests

from fantasy.config import settings
from fantasy.notify.base import render_text
from fantasy.orchestrator.models import Proposal, ProposalKind

log = logging.getLogger(__name__)


class NtfyNotifier:
    def __init__(self):
        self.topic = settings.ntfy_topic
        self.base = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")

    def notify(self, proposal: Proposal) -> str | None:
        headers = {"Title": proposal.title, "Tags": proposal.kind.value}
        if proposal.kind != ProposalKind.alert and self.base:
            headers["Actions"] = (
                f"http, ✅ Approve, {self.base}/proposals/{proposal.id}/approve, method=POST; "
                f"http, ❌ Reject, {self.base}/proposals/{proposal.id}/reject, method=POST"
            )
        try:
            requests.post(f"https://ntfy.sh/{self.topic}", data=render_text(proposal).encode(),
                          headers=headers, timeout=10)
            return f"ntfy:{proposal.id}"
        except Exception as e:  # noqa: BLE001
            log.warning("ntfy notify failed: %s", e)
            return None
