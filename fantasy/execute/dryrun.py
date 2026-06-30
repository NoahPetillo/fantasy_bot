"""Dry-run executor — simulates a write for testing the approve→execute flow."""

from __future__ import annotations

import logging

from fantasy.execute.base import ExecutionResult
from fantasy.orchestrator.models import Proposal

log = logging.getLogger(__name__)


class DryRunExecutor:
    name = "dryrun"

    def execute(self, proposal: Proposal) -> ExecutionResult:
        log.info("[DRYRUN] would execute %s: %s | payload=%s",
                 proposal.kind.value, proposal.title, proposal.payload)
        return ExecutionResult(
            ok=True, backend=self.name, ref=f"dryrun:{proposal.id}",
            message=f"[dry-run] simulated {proposal.kind.value}: {proposal.title}",
            performed=dict(proposal.payload),
        )
