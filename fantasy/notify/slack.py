"""Slack notifier — Block Kit message with Approve/Reject buttons.

Uses the bot token to post to the configured channel. The buttons carry the
proposal id in their ``value`` and ``action_id`` so the FastAPI interactions
endpoint (fantasy.api.app) can resolve the click back to the proposal. Requires
``slack_sdk`` (install the ``notify`` extra). Socket Mode for receiving clicks is
wired in the API layer.
"""

from __future__ import annotations

import logging

from fantasy.config import settings
from fantasy.notify.base import render_text
from fantasy.orchestrator.models import Proposal, ProposalKind

log = logging.getLogger(__name__)


def _blocks(p: Proposal) -> list[dict]:
    blocks = [
        {"type": "section", "text": {"type": "mrkdwn",
         "text": f"*{p.title}*\n{p.detail}"}},
        {"type": "context", "elements": [{"type": "mrkdwn",
         "text": f"{p.kind.value} · value {p.value:+.1f} · conf {p.confidence*100:.0f}% · wk {p.week}"}]},
    ]
    if p.kind != ProposalKind.alert:
        blocks.append({
            "type": "actions",
            "block_id": f"prop_{p.id}",
            "elements": [
                {"type": "button", "style": "primary", "text": {"type": "plain_text", "text": "✅ Approve"},
                 "action_id": "approve_proposal", "value": p.id},
                {"type": "button", "style": "danger", "text": {"type": "plain_text", "text": "❌ Reject"},
                 "action_id": "reject_proposal", "value": p.id},
            ],
        })
    return blocks


class SlackNotifier:
    def __init__(self):
        from slack_sdk import WebClient

        self.client = WebClient(token=settings.slack_bot_token)
        self.channel = settings.slack_channel_id

    def notify(self, proposal: Proposal) -> str | None:
        try:
            resp = self.client.chat_postMessage(
                channel=self.channel, blocks=_blocks(proposal), text=proposal.title
            )
            return f"slack:{resp['ts']}"
        except Exception as e:  # noqa: BLE001
            log.warning("Slack notify failed: %s", e)
            return None
