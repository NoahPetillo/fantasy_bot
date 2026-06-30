"""Expert registry — loaded from config/experts.yaml (edit without code changes)."""

from __future__ import annotations

import functools
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class Expert(BaseModel):
    name: str
    x_handle: str | None = None
    bsky_handle: str | None = None
    outlet: str = "Independent"
    group: str = "waiver"
    tier: int = 2
    base_trust: float = 0.7
    specialties: list[str] = Field(default_factory=list)


class RssFeed(BaseModel):
    name: str
    url: str
    outlet: str = "Independent"
    base_trust: float = 0.7


class Registry(BaseModel):
    experts: list[Expert] = Field(default_factory=list)
    rss_feeds: list[RssFeed] = Field(default_factory=list)

    def with_bluesky(self) -> list[Expert]:
        return [e for e in self.experts if e.bsky_handle]

    def by_handle(self) -> dict[str, Expert]:
        out = {}
        for e in self.experts:
            for h in (e.x_handle, e.bsky_handle):
                if h:
                    out[h] = e
        return out


_DEFAULT = Path(__file__).resolve().parents[3] / "config" / "experts.yaml"


@functools.lru_cache(maxsize=4)
def load_registry(path: str | None = None) -> Registry:
    p = Path(path) if path else _DEFAULT
    data = yaml.safe_load(p.read_text()) if p.exists() else {}
    return Registry(experts=data.get("experts", []), rss_feeds=data.get("rss_feeds", []))
