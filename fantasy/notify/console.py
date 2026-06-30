"""Console notifier — used in dry-run and as a safe default."""

from __future__ import annotations

from fantasy.notify.base import approve_reject_hint, render_text
from fantasy.orchestrator.models import Proposal


class ConsoleNotifier:
    def notify(self, proposal: Proposal) -> str | None:
        print("\n" + "─" * 72)
        print(render_text(proposal))
        print(approve_reject_hint(proposal))
        return f"console:{proposal.id}"
