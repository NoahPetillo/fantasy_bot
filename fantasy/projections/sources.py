"""External free projection sources for the multi-source consensus.

Wisdom-of-crowds beats any single source (incl. ESPN) year over year, so the
projection backbone averages several diverse, free, forward-looking sources.
Each source returns gsis_id -> projected points in the league's format.
"""

from __future__ import annotations

import logging

import pandas as pd
import requests

from fantasy.config import settings
from fantasy.data.ids import crosswalk
from fantasy.league_settings import LeagueSettings, ScoringFormat

log = logging.getLogger(__name__)

_SLEEPER = "https://api.sleeper.com/projections/nfl/{season}/{week}?season_type=regular&order_by=pts_ppr"
_FMT_KEY = {ScoringFormat.ppr: "pts_ppr", ScoringFormat.half_ppr: "pts_half_ppr",
            ScoringFormat.standard: "pts_std"}


class SleeperProjectionSource:
    """Sleeper's free weekly projections, scored in the league's format."""

    name = "sleeper"

    def weekly_points(self, season: int, week: int, league: LeagueSettings) -> dict[str, float]:
        key = _FMT_KEY.get(league.scoring_format, "pts_ppr")
        path = settings.cache_dir / f"sleeperproj_{season}_{week}.parquet"
        try:
            df = pd.read_parquet(path)
        except (FileNotFoundError, OSError):
            try:
                r = requests.get(_SLEEPER.format(season=season, week=week),
                                 headers={"User-Agent": "fantasy-app/0.1"}, timeout=25)
                r.raise_for_status()
                rows = [{"sleeper_id": str(it.get("player_id")),
                         "pts": (it.get("stats") or {}).get(key)} for it in r.json()]
                df = pd.DataFrame(rows).dropna(subset=["pts"])
                df.to_parquet(path, index=False)
            except Exception as e:  # noqa: BLE001
                log.warning("Sleeper projections failed (%s); source skipped.", e)
                return {}
        xw = crosswalk()
        out: dict[str, float] = {}
        for r in df.itertuples(index=False):
            gid = xw.from_sleeper(r.sleeper_id)
            if gid:
                out[gid] = float(r.pts)
        return out
