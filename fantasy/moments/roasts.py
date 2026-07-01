"""Roast book — optional, per-league inside jokes sprinkled into captions.

Loaded from ``config/roasts.yaml`` (path configurable). The block matching the
current ``ESPN_LEAGUE_ID`` is used, so switching leagues each season just means
adding a new block. A given manager's joke fires only ~1 in
``content_roast_frequency`` of the moments they star in (deterministic by
league+manager+week, so it's stable per run but varies week to week) — that
keeps the bits feeling occasional instead of every-single-week.

Everything degrades to "no joke" on a missing file / bad YAML / unknown league.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from fantasy.moments.config import content_config as settings  # decoupled from the app

log = logging.getLogger(__name__)

_cache: dict | None = None


def _resolve_path() -> Path:
    p = Path(settings.content_roasts_file)
    if p.exists():
        return p
    # Fall back to repo-root-relative (cwd-independent).
    root = Path(__file__).resolve().parents[2]
    return root / p


def _load() -> dict:
    global _cache
    if _cache is not None:
        return _cache
    try:
        import yaml

        path = _resolve_path()
        _cache = yaml.safe_load(path.read_text()) or {} if path.exists() else {}
    except Exception as e:  # noqa: BLE001
        log.info("Roast book unavailable (%s); no inside jokes.", e)
        _cache = {}
    return _cache


def reset_cache() -> None:
    """Forget the loaded file (used by tests after pointing at a new path)."""
    global _cache
    _cache = None


def _league_managers() -> list[dict]:
    entry = (_load().get("leagues", {}) or {}).get(str(settings.espn_league_id), {}) or {}
    return entry.get("managers", []) or []


def roast_for(manager: str | None, week: int) -> str | None:
    """An inside-joke line for ``manager`` this week, or None (most of the time)."""
    if not manager:
        return None
    rec = next((m for m in _league_managers()
                if str(m.get("name", "")).strip().lower() == manager.strip().lower()), None)
    if not rec:
        return None
    notes = rec.get("notes") or []
    if not notes:
        return None
    freq = max(1, int(settings.content_roast_frequency))
    seed = f"{settings.espn_league_id}:{manager.lower()}:{week}"
    h = int(hashlib.sha1(seed.encode()).hexdigest(), 16)
    if h % freq != 0:
        return None  # not this week — keep it occasional
    note = notes[h % len(notes)]
    nick = rec.get("nickname")
    if nick and str(nick).strip().lower() != manager.strip().lower():
        return f"(goes by '{nick}') {note}"
    return note
