"""RSS adapter — free fantasy news/analysis feeds (RotoWire, Rotoworld, Substacks)."""

from __future__ import annotations

import logging

import feedparser

from fantasy.news.experts.models import RawPost
from fantasy.news.experts.registry import Registry

log = logging.getLogger(__name__)


class RssAdapter:
    name = "rss"

    def fetch(self, registry: Registry, limit_per_author: int = 30) -> list[RawPost]:
        posts: list[RawPost] = []
        for feed in registry.rss_feeds:
            try:
                parsed = feedparser.parse(feed.url)
                for entry in parsed.entries[:limit_per_author]:
                    title = entry.get("title", "")
                    summary = entry.get("summary", "")
                    text = f"{title}. {summary}".strip()
                    if not text:
                        continue
                    posts.append(RawPost(
                        id=entry.get("id", entry.get("link", title)),
                        author_handle=feed.name, outlet=feed.outlet, platform="rss",
                        text=text, ts=entry.get("published"), url=entry.get("link"),
                        base_trust=feed.base_trust,
                    ))
            except Exception as ex:  # noqa: BLE001
                log.warning("RSS fetch failed for %s: %s", feed.url, ex)
        log.info("RSS: %d entries from %d feeds", len(posts), len(registry.rss_feeds))
        return posts
