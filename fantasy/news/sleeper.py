"""Sleeper ingester — free crowd signal (trending add/drop) + structured injuries.

Sleeper's player file carries clean ``injury_status`` designations (Out / Doubtful
/ Questionable / IR / PUP), which is a reliable, no-LLM-needed signal. Trending
add/drop is the best free waiver-momentum signal. Both map to gsis ids via the
crosswalk so they line up with the projection board.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import requests

from fantasy.config import settings
from fantasy.data.ids import crosswalk
from fantasy.news.models import EventType, PlayerSignal

log = logging.getLogger(__name__)

BASE = "https://api.sleeper.app/v1"
UA = {"User-Agent": "fantasy-app/0.1"}
_INJURY_MAP = {
    "Out": EventType.injury_out, "IR": EventType.ir, "Doubtful": EventType.injury_doubtful,
    "Questionable": EventType.injury_questionable, "PUP": EventType.ir, "Sus": EventType.injury_out,
}


def fetch_trending(kind: str = "add", limit: int = 25, lookback_hours: int = 24) -> list[tuple[str, int]]:
    url = f"{BASE}/players/nfl/trending/{kind}?lookback_hours={lookback_hours}&limit={limit}"
    r = requests.get(url, headers=UA, timeout=20)
    r.raise_for_status()
    return [(d["player_id"], d.get("count", 0)) for d in r.json()]


def fetch_player_index(refresh: bool = False, max_age_hours: int = 24) -> dict:
    """The full Sleeper player file (cached to disk; refreshed at most daily)."""
    path: Path = settings.cache_dir / "sleeper_players.json"
    if not refresh and path.exists() and (time.time() - path.stat().st_mtime) < max_age_hours * 3600:
        return json.loads(path.read_text())
    log.info("Downloading Sleeper player file (~5MB)...")
    r = requests.get(f"{BASE}/players/nfl", headers=UA, timeout=60)
    r.raise_for_status()
    data = r.json()
    path.write_text(json.dumps(data))
    return data


def trending_signals(limit: int = 25) -> list[PlayerSignal]:
    xw = crosswalk()
    out: list[PlayerSignal] = []
    for kind, etype in (("add", EventType.trending_add), ("drop", EventType.trending_drop)):
        for sid, count in fetch_trending(kind, limit=limit):
            gid = xw.from_sleeper(sid)
            out.append(PlayerSignal(
                player_id=gid, player_name=xw.name(gid) if gid else sid,
                position=xw.gsis_to_pos.get(gid) if gid else None,
                event_type=etype, source="sleeper",
                summary=f"{count:,} managers {kind}ed in last 24h",
                confidence=min(0.5 + count / 20000, 0.95),
            ))
    return out


def injury_signals(player_ids: set[str] | None = None) -> list[PlayerSignal]:
    """Injury-status signals; if ``player_ids`` given, restrict to those gsis ids."""
    xw = crosswalk()
    idx = fetch_player_index()
    out: list[PlayerSignal] = []
    for sid, meta in idx.items():
        status = meta.get("injury_status")
        if not status or status not in _INJURY_MAP:
            continue
        gid = xw.from_sleeper(sid)
        if player_ids is not None and gid not in player_ids:
            continue
        name = meta.get("full_name") or (xw.name(gid) if gid else sid)
        out.append(PlayerSignal(
            player_id=gid, player_name=name, position=meta.get("position"),
            team=meta.get("team"), event_type=_INJURY_MAP[status], source="sleeper",
            summary=f"Injury status: {status}", confidence=0.9,
        ))
    return out
