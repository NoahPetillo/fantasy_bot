"""Bluesky adapter — free, no-auth public reads (AT Protocol AppView).

Pulls recent posts for every registry expert that has a ``bsky_handle``. No API
key, no cost. Handles that don't resolve are skipped (logged once).
"""

from __future__ import annotations

import logging

import requests

from fantasy.news.experts.models import RawPost
from fantasy.news.experts.registry import Registry

log = logging.getLogger(__name__)

BASE = "https://public.api.bsky.app/xrpc"


class BlueskyAdapter:
    name = "bluesky"

    def fetch(self, registry: Registry, limit_per_author: int = 10) -> list[RawPost]:
        posts: list[RawPost] = []
        for e in registry.with_bluesky():
            try:
                r = requests.get(
                    f"{BASE}/app.bsky.feed.getAuthorFeed",
                    params={"actor": e.bsky_handle, "limit": limit_per_author,
                            "filter": "posts_no_replies"},
                    timeout=15,
                )
                if r.status_code != 200:
                    log.info("Bluesky %s -> %s (skip)", e.bsky_handle, r.status_code)
                    continue
                for item in r.json().get("feed", []):
                    post = item.get("post", {})
                    rec = post.get("record", {})
                    text = rec.get("text", "")
                    if not text:
                        continue
                    uri = post.get("uri", "")
                    posts.append(RawPost(
                        id=uri or f"{e.bsky_handle}:{rec.get('createdAt','')}",
                        author_handle=e.bsky_handle, outlet=e.outlet, platform="bluesky",
                        text=text, ts=rec.get("createdAt"), base_trust=e.base_trust,
                        url=_post_url(e.bsky_handle, uri),
                    ))
            except Exception as ex:  # noqa: BLE001
                log.warning("Bluesky fetch failed for %s: %s", e.bsky_handle, ex)
        log.info("Bluesky: %d posts from %d experts", len(posts), len(registry.with_bluesky()))
        return posts


def _post_url(handle: str, uri: str) -> str | None:
    rkey = uri.rsplit("/", 1)[-1] if uri else ""
    return f"https://bsky.app/profile/{handle}/post/{rkey}" if rkey else None
