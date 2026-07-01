"""Test-suite defaults that keep runs hermetic regardless of the developer's
local ``.env``.

Most importantly: the shared-password gate is forced OFF by default. The endpoint
tests don't authenticate, so if a real ``SITE_PASSWORD`` is set in ``.env`` (e.g.
you turned the gate on for your deployment) those tests would otherwise 401. Tests
that actually exercise the gate (``test_auth_gate``, ``test_chat_ratelimit``) opt
back in by monkeypatching the password in their own fixtures, which runs after
this autouse fixture and so wins during the test.
"""

from __future__ import annotations

import pytest

import fantasy.api.app as api
from fantasy.config import settings


@pytest.fixture(autouse=True)
def _hermetic_web_settings(monkeypatch):
    # Gate off unless a test explicitly turns it on.
    monkeypatch.setattr(settings, "site_password", None)
    # Fresh chat limiter per test so a tiny per-test limit can't leak across tests.
    api._chat_limiter = None
    yield
    api._chat_limiter = None


@pytest.fixture
def db(tmp_path):
    """An isolated per-test database (SQLite file) with the full schema created,
    yielding a session. Points the whole data layer at this DB so anything using
    ``get_db``/``get_sessionmaker`` (e.g. the API) hits the same isolated store."""
    from fantasy.db import base, create_all, reset_engine

    base.configure_engine(f"sqlite:///{tmp_path / 'test.sqlite'}")
    create_all()
    session = base.get_sessionmaker()()
    try:
        yield session
    finally:
        session.close()
        reset_engine()
