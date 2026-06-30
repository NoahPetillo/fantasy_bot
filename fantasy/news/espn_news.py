"""ESPN public news ingester.

ESPN's free news endpoint embeds athlete + team references in each article, so
headlines map directly onto rostered players. We apply a deterministic
injury-keyword classifier as a baseline; the LLM extractor (extract.py) enriches
unstructured articles with beneficiaries when an API key is available.
"""

from __future__ import annotations

import logging

import requests

from fantasy.data.ids import crosswalk
from fantasy.news.models import EventType, PlayerSignal

log = logging.getLogger(__name__)

NEWS_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/news"
UA = {"User-Agent": "Mozilla/5.0"}

_KEYWORDS = [
    (("ruled out", "will not play", "won't play", "inactive"), EventType.injury_out),
    (("doubtful",), EventType.injury_doubtful),
    (("questionable", "limited"), EventType.injury_questionable),
    (("placed on ir", "injured reserve", " to ir"), EventType.ir),
    (("returns", "activated", "cleared", "will play", "back at practice"), EventType.injury_return),
    (("traded", "trade", "acquired"), EventType.trade),
    (("starter", "promoted", "depth chart", "benched", "demoted"), EventType.depth_change),
]


def fetch_news(limit: int = 50) -> list[dict]:
    r = requests.get(NEWS_URL, params={"limit": limit}, headers=UA, timeout=20)
    r.raise_for_status()
    return r.json().get("articles", [])


def _classify(text: str) -> EventType:
    t = text.lower()
    for words, etype in _KEYWORDS:
        if any(w in t for w in words):
            return etype
    return EventType.news


def news_signals(limit: int = 50) -> list[PlayerSignal]:
    xw = crosswalk()
    out: list[PlayerSignal] = []
    for art in fetch_news(limit):
        head = art.get("headline", "")
        desc = art.get("description", "")
        url = (art.get("links", {}).get("web", {}) or {}).get("href")
        pub = art.get("published")
        etype = _classify(f"{head} {desc}")
        athletes = [c for c in art.get("categories", []) if c.get("type") == "athlete"]
        if not athletes:
            continue
        a = athletes[0]
        gid = xw.from_espn(a.get("athlete", {}).get("id") or a.get("id"))
        out.append(PlayerSignal(
            player_id=gid, player_name=a.get("description", "") or (xw.name(gid) if gid else "?"),
            position=xw.gsis_to_pos.get(gid) if gid else None,
            event_type=etype, source="espn", summary=head, source_url=url, published_ts=pub,
            confidence=0.75 if etype != EventType.news else 0.4,
        ))
    return out
