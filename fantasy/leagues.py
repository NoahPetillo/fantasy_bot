"""Multi-league registry.

The user's ESPN cookies (``espn_s2``/``SWID``) authenticate their whole ESPN
account, so adding another league needs only its ``league_id`` + ``team_id`` (+
season) — no new login. Registered leagues are stored as a small JSON file in the
data dir; each gets its own dashboard snapshot (``dashboard_<league_id>.json``).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from fantasy.config import settings

log = logging.getLogger(__name__)


@dataclass
class LeagueRef:
    league_id: int
    team_id: int | None = None
    season: int = 2025
    name: str = ""
    added_at: str = ""

    @property
    def key(self) -> str:
        return str(self.league_id)


class LeagueRegistry:
    """JSON-backed list of the user's leagues. Upsert by ``league_id``."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else settings.data_dir / "leagues.json"

    def _read(self) -> list[dict]:
        try:
            return json.loads(self.path.read_text())
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return []

    def _write(self, items: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(items, indent=2))

    def all(self) -> list[LeagueRef]:
        return [LeagueRef(**d) for d in self._read()]

    def get(self, league_id: int | str) -> LeagueRef | None:
        lid = int(league_id)
        return next((r for r in self.all() if r.league_id == lid), None)

    def add(self, ref: LeagueRef) -> LeagueRef:
        if not ref.added_at:
            ref.added_at = datetime.now(timezone.utc).isoformat()
        items = [d for d in self._read() if int(d.get("league_id")) != ref.league_id]
        items.append(asdict(ref))
        self._write(items)
        log.info("Registered league %s (%s)", ref.league_id, ref.name or "?")
        return ref

    def remove(self, league_id: int | str) -> bool:
        lid = int(league_id)
        items = self._read()
        kept = [d for d in items if int(d.get("league_id")) != lid]
        if len(kept) == len(items):
            return False
        self._write(kept)
        return True

    def seed_default(self) -> LeagueRef | None:
        """Bootstrap the registry from .env on first run so the existing league
        shows up without any manual add."""
        if self.all():
            return None
        if not settings.espn_league_id:
            return None
        return self.add(LeagueRef(
            league_id=settings.espn_league_id, team_id=settings.espn_team_id,
            season=settings.espn_season, name="",
        ))


# Module-level default registry + convenience wrappers.
_registry = LeagueRegistry()


def registry() -> LeagueRegistry:
    return _registry
