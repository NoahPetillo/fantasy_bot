"""X/Twitter adapter — official API v2, OFF by default behind a feature flag.

Enable with ENABLE_X_SOURCE=true + X_BEARER_TOKEN (official pay-per-use; ToS-
compliant inference, NO scraping). Reads only the curated tier-1 X-only breakers
to keep cost minimal. When disabled (default), the free Bluesky+RSS stack is used.
"""

from __future__ import annotations

import logging

import requests

from fantasy.config import settings
from fantasy.news.experts.models import RawPost
from fantasy.news.experts.registry import Registry

log = logging.getLogger(__name__)

API = "https://api.twitter.com/2"


class XAdapter:
    name = "x"

    @staticmethod
    def enabled() -> bool:
        return bool(settings.enable_x_source and settings.x_bearer_token)

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {settings.x_bearer_token}"}

    def fetch(self, registry: Registry, limit_per_author: int = 5,
              only_tier: int = 1) -> list[RawPost]:
        if not self.enabled():
            return []
        posts: list[RawPost] = []
        # Cost control: only the X-only breakers we can't get free, top tier only.
        targets = [e for e in registry.experts
                   if e.x_handle and e.tier <= only_tier and not e.bsky_handle]
        for e in targets:
            try:
                u = requests.get(f"{API}/users/by/username/{e.x_handle}",
                                 headers=self._headers(), timeout=15)
                if u.status_code != 200:
                    continue
                uid = u.json()["data"]["id"]
                t = requests.get(
                    f"{API}/users/{uid}/tweets",
                    headers=self._headers(),
                    params={"max_results": max(limit_per_author, 5),
                            "tweet.fields": "created_at", "exclude": "retweets,replies"},
                    timeout=15,
                )
                if t.status_code != 200:
                    log.info("X %s -> %s", e.x_handle, t.status_code)
                    continue
                for tw in t.json().get("data", []):
                    posts.append(RawPost(
                        id=tw["id"], author_handle=e.x_handle, outlet=e.outlet, platform="x",
                        text=tw.get("text", ""), ts=tw.get("created_at"),
                        url=f"https://x.com/{e.x_handle}/status/{tw['id']}", base_trust=e.base_trust,
                    ))
            except Exception as ex:  # noqa: BLE001
                log.warning("X fetch failed for %s: %s", e.x_handle, ex)
        log.info("X: %d posts from %d breakers", len(posts), len(targets))
        return posts
