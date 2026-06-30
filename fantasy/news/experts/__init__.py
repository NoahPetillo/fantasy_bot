"""Expert-signal layer — the differentiator: curated experts → real-time posts →
trust-weighted, corroboration-gated nudges to the decision stack.

Pipeline: adapters (Bluesky/RSS/X) → extract (LLM/keyword) → fuse (+gate) → adjust.
"""

from __future__ import annotations

import logging

from fantasy.news.experts.extract import extract_signals
from fantasy.news.experts.fusion import fuse
from fantasy.news.experts.models import ExpertSignal, FusedSignal, RawPost
from fantasy.news.experts.registry import Registry, load_registry

log = logging.getLogger(__name__)


def default_adapters():
    from fantasy.news.experts.sources.bluesky import BlueskyAdapter
    from fantasy.news.experts.sources.rss import RssAdapter
    from fantasy.news.experts.sources.x import XAdapter

    adapters = [BlueskyAdapter(), RssAdapter()]
    if XAdapter.enabled():  # OFF unless ENABLE_X_SOURCE=true + X_BEARER_TOKEN set
        adapters.append(XAdapter())
    return adapters


def gather_posts(registry: Registry | None = None, adapters=None) -> list[RawPost]:
    registry = registry or load_registry()
    adapters = adapters if adapters is not None else default_adapters()
    posts: list[RawPost] = []
    for a in adapters:
        try:
            posts += a.fetch(registry)
        except Exception as e:  # noqa: BLE001
            log.warning("adapter %s failed: %s", getattr(a, "name", a), e)
    return posts


def collect_signals(registry: Registry | None = None, adapters=None) -> list[FusedSignal]:
    """End-to-end: posts → extracted signals → fused, gated signals."""
    return fuse(extract_signals(gather_posts(registry, adapters)))


__all__ = ["ExpertSignal", "FusedSignal", "RawPost", "Registry", "load_registry",
           "gather_posts", "collect_signals", "extract_signals", "fuse"]
