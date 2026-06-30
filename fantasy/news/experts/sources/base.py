"""Source adapter interface — X / Bluesky / RSS are swappable behind this."""

from __future__ import annotations

from typing import Protocol

from fantasy.news.experts.models import RawPost
from fantasy.news.experts.registry import Registry


class SourceAdapter(Protocol):
    name: str

    def fetch(self, registry: Registry, limit_per_author: int = 10) -> list[RawPost]:
        """Return recent posts from the experts this adapter can reach."""
        ...
