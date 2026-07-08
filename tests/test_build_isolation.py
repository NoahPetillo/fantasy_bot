"""Isolated-build supervisor logic (the durable fix for build OOMs).

Heavy builds run in a spawned subprocess in production so an out-of-memory kill
takes down only that build, not the web server. These tests cover the parent-side
supervisor that turns a child's outcome — a status file, a clean exit, or a
SIGKILL from the OS OOM-killer — into the status string the GET endpoints poll.
They exercise the logic directly with a fake process (no real spawning), so they
stay fast and deterministic; the real spawn path is verified manually.
"""

from __future__ import annotations

import time

import fantasy.api.app as api


class _FakeProc:
    def __init__(self, exitcode: int):
        self.exitcode = exitcode

    def join(self):  # already "finished"
        return None


def test_supervise_reports_oom_when_child_killed(tmp_path):
    """No status file + negative exit code = SIGKILL (OOM) -> clear OOM message."""
    status: dict[str, str] = {}
    api._supervise_process(_FakeProc(-9), "k", status, tmp_path / "s")
    assert "out of memory" in status["k"].lower()


def test_supervise_uses_status_file_when_present(tmp_path):
    path = tmp_path / "s"
    path.write_text("done", encoding="utf-8")
    status: dict[str, str] = {}
    api._supervise_process(_FakeProc(0), "k", status, path)
    assert status["k"] == "done"
    assert not path.exists()  # cleaned up


def test_supervise_reports_unexpected_nonzero_exit(tmp_path):
    status: dict[str, str] = {}
    api._supervise_process(_FakeProc(1), "k", status, tmp_path / "s")
    assert "unexpectedly" in status["k"] and "1" in status["k"]


def test_launch_build_thread_mode_reflects_worker_status(tmp_path, monkeypatch):
    """With build_subprocess off (conftest default for tests), the worker runs in
    a thread and its file-written status is copied back into the dict."""
    monkeypatch.setattr(api.settings, "data_dir", tmp_path)

    def worker(user_id, league_id, status_path):
        api.write_build_status(status_path, "done")

    status: dict[str, str] = {}
    api._launch_build("full", "k", worker, ("u", "l"), status)
    deadline = time.time() + 5
    while time.time() < deadline and status.get("k") in (None, "building"):
        time.sleep(0.02)
    assert status["k"] == "done"


def test_launch_build_thread_mode_worker_crash(tmp_path, monkeypatch):
    """A worker that raises without writing a status still resolves (not stuck)."""
    monkeypatch.setattr(api.settings, "data_dir", tmp_path)

    def worker(status_path):
        raise RuntimeError("boom")

    status: dict[str, str] = {}
    api._launch_build("plan", "k", worker, (), status)
    deadline = time.time() + 5
    while time.time() < deadline and status.get("k") in (None, "building"):
        time.sleep(0.02)
    assert status["k"].startswith("error")
