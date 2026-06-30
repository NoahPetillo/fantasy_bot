"""Execution layer: deep-link generation + the approve→execute safety gates."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from fantasy.config import ExecutionBackend, ExecutionMode, settings
from fantasy.execute.base import execute_approved, get_executor
from fantasy.execute.deeplink import DeepLinkExecutor
from fantasy.execute.dryrun import DryRunExecutor
from fantasy.orchestrator.models import Proposal, ProposalKind, ProposalStatus
from fantasy.orchestrator.store import Store


@pytest.fixture
def restore_mode():
    mode, backend = settings.execution_mode, settings.execution_backend
    yield
    settings.execution_mode, settings.execution_backend = mode, backend


def _prop(kind=ProposalKind.waiver):
    return Proposal(kind=kind, season=2024, week=6, team_id=3, title="t",
                    payload={"add": "A", "drop": "B", "faab_bid": 12,
                             "key_fields": {"add": "A", "drop": "B"}})


def test_deeplink_builds_urls_per_kind():
    ex = DeepLinkExecutor()
    for kind in (ProposalKind.start_sit, ProposalKind.waiver, ProposalKind.trade):
        r = ex.execute(_prop(kind))
        assert r.ok and r.ref and r.ref.startswith("https://fantasy.espn.com/football")


def test_advise_mode_never_writes(restore_mode):
    settings.execution_mode = ExecutionMode.advise
    store = Store(Path(tempfile.mkdtemp()) / "e.sqlite")
    p = _prop()
    store.add(p)
    r = execute_approved(p, store, DryRunExecutor())
    assert r.ok is False and "advise" in r.message
    assert not store.has_executed(p.idempotency_key)


def test_approve_mode_executes_and_is_idempotent(restore_mode):
    settings.execution_mode = ExecutionMode.approve
    store = Store(Path(tempfile.mkdtemp()) / "e.sqlite")
    p = _prop()
    store.add(p)

    r1 = execute_approved(p, store, DryRunExecutor())
    assert r1.ok and store.get(p.id).status == ProposalStatus.executed
    assert store.has_executed(p.idempotency_key)

    # Second execution is a guarded no-op (no double-submit).
    r2 = execute_approved(p, store, DryRunExecutor())
    assert r2.ok and "already executed" in r2.message


def test_failure_marks_failed(restore_mode):
    settings.execution_mode = ExecutionMode.approve
    store = Store(Path(tempfile.mkdtemp()) / "e.sqlite")
    p = _prop()
    store.add(p)

    class Boom:
        name = "boom"

        def execute(self, proposal):
            raise RuntimeError("kaboom")

    r = execute_approved(p, store, Boom())
    assert r.ok is False and "kaboom" in r.message
    assert store.get(p.id).status == ProposalStatus.failed


def test_get_executor_respects_backend(restore_mode):
    settings.execution_backend = ExecutionBackend.deeplink
    assert get_executor().name == "deeplink"
    settings.execution_backend = ExecutionBackend.dryrun
    assert get_executor().name == "dryrun"
