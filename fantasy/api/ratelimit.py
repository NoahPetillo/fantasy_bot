"""A tiny in-process rate limiter for the public chatbot.

The chatbot endpoint is open (league mates use it without the password), so a
single client — or a crawler that finds the URL — could spam it. This caps
questions per client IP over a sliding window. It's deliberately simple: an
in-memory sliding-window counter, no Redis. The app runs as a single uvicorn
process, so per-process state is the whole picture; if you ever scale to multiple
workers/instances the limit becomes per-worker (still fine for a generous
abuse-prevention cap, just not exact).
"""

from __future__ import annotations

import threading
import time
from collections import deque

from fastapi import Request


class RateLimiter:
    """Sliding-window limiter: at most ``limit`` hits per ``window`` seconds per
    key. ``limit <= 0`` disables it (always allowed)."""

    def __init__(self, limit: int, window: int) -> None:
        self.limit = limit
        self.window = window
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()
        self._last_sweep = 0.0

    def check(self, key: str) -> tuple[bool, int]:
        """Record a hit for ``key``. Returns ``(allowed, retry_after_seconds)``;
        when blocked, ``retry_after`` is when the oldest hit ages out."""
        if self.limit <= 0:
            return True, 0
        now = time.time()
        with self._lock:
            self._maybe_sweep(now)
            dq = self._hits.setdefault(key, deque())
            cutoff = now - self.window
            while dq and dq[0] <= cutoff:
                dq.popleft()
            if len(dq) >= self.limit:
                return False, max(1, int(dq[0] + self.window - now) + 1)
            dq.append(now)
            return True, 0

    def _maybe_sweep(self, now: float) -> None:
        """Drop expired timestamps and idle keys so memory stays bounded to the
        set of recently-active IPs. Runs at most once per window."""
        if now - self._last_sweep < self.window:
            return
        self._last_sweep = now
        cutoff = now - self.window
        for k in list(self._hits):
            dq = self._hits[k]
            while dq and dq[0] <= cutoff:
                dq.popleft()
            if not dq:
                del self._hits[k]


def client_ip(request: Request) -> str:
    """Best-effort client IP. Behind Render/Railway the real client is in the
    ``X-Forwarded-For`` header (first entry); fall back to the socket peer. Note a
    client can spoof XFF to dodge the limit — acceptable for a generous abuse cap,
    not a security control."""
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"
