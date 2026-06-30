"""Execution layer — carry out an APPROVED proposal (Phase 3).

The design keeps decision and execution decoupled and SWAPPABLE: the same
approved proposal can be fulfilled by a deep-link (safe default), a Playwright
browser write (ESPN), or a dry-run simulation — or, later, a Yahoo official-API
backend. Every execution passes through ``execute_approved``, which enforces:

1. mode gate  — never executes in ``advise`` mode.
2. idempotency — never executes the same action twice (checks the action log).
"""

from __future__ import annotations

import logging
from typing import Protocol

from pydantic import BaseModel

from fantasy.config import ExecutionBackend, ExecutionMode, settings
from fantasy.orchestrator.models import Proposal, ProposalStatus
from fantasy.orchestrator.store import Store

log = logging.getLogger(__name__)


class ExecutionResult(BaseModel):
    ok: bool
    backend: str
    message: str = ""
    ref: str | None = None  # url (deep-link) or transaction id
    performed: dict = {}


class Executor(Protocol):
    name: str

    def execute(self, proposal: Proposal) -> ExecutionResult: ...


def get_executor(backend: ExecutionBackend | None = None) -> Executor:
    backend = backend or settings.execution_backend
    if backend == ExecutionBackend.playwright:
        from fantasy.execute.playwright_tier import PlaywrightExecutor

        return PlaywrightExecutor()
    if backend == ExecutionBackend.dryrun:
        from fantasy.execute.dryrun import DryRunExecutor

        return DryRunExecutor()
    from fantasy.execute.deeplink import DeepLinkExecutor

    return DeepLinkExecutor()


def execute_approved(
    proposal: Proposal, store: Store, executor: Executor | None = None
) -> ExecutionResult:
    """Execute an approved proposal under the safety gates. Idempotent."""
    if settings.execution_mode == ExecutionMode.advise:
        return ExecutionResult(ok=False, backend="none",
                               message="advise mode — no writes performed")
    if store.has_executed(proposal.idempotency_key):
        log.info("Already executed %s — skipping (idempotent).", proposal.idempotency_key)
        return ExecutionResult(ok=True, backend="none", message="already executed (no-op)")

    executor = executor or get_executor()
    try:
        result = executor.execute(proposal)
    except Exception as e:  # noqa: BLE001
        log.exception("Execution failed for %s", proposal.id)
        store.set_status(proposal.id, ProposalStatus.failed)
        return ExecutionResult(ok=False, backend=getattr(executor, "name", "?"), message=str(e))

    store.set_status(
        proposal.id,
        ProposalStatus.executed if result.ok else ProposalStatus.failed,
        notify_ref=result.ref,
    )
    return result
