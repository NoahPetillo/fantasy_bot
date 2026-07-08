"""External free projection sources for the multi-source consensus.

Wisdom-of-crowds beats any single source (incl. ESPN) year over year, so the
projection backbone averages several diverse, free, forward-looking sources.
Each source returns gsis_id -> projected points in the league's format.
"""

from __future__ import annotations

import logging
import time

import pandas as pd
import requests

from fantasy.config import settings
from fantasy.data.ids import crosswalk
from fantasy.league_settings import LeagueSettings, ScoringFormat

log = logging.getLogger(__name__)

_SLEEPER = "https://api.sleeper.com/projections/nfl/{season}/{week}?season_type=regular&order_by=pts_ppr"
_SLEEPER_SEASON = "https://api.sleeper.com/projections/nfl/{season}?season_type=regular"
# Season-projection cache TTL — preseason numbers move as sources update them.
_SEASON_PROJ_TTL_SECONDS = 24 * 3600
_FMT_KEY = {ScoringFormat.ppr: "pts_ppr", ScoringFormat.half_ppr: "pts_half_ppr",
            ScoringFormat.standard: "pts_std"}

# Sleeper season-projection stat key -> our canonical stat name. Offense + kicking
# where trivially mappable, and IDP (Sleeper is our only real IDP projection
# source). Keys verified against the live API; absent keys are simply skipped.
_SLEEPER_STAT_MAP: dict[str, str] = {
    # Passing
    "pass_att": "pass_attempts", "pass_cmp": "pass_completions",
    "pass_yd": "passing_yards", "pass_td": "passing_tds",
    "pass_int": "passing_interceptions", "pass_2pt": "passing_2pt_conversions",
    # Rushing
    "rush_att": "rushing_attempts", "rush_yd": "rushing_yards",
    "rush_td": "rushing_tds", "rush_2pt": "rushing_2pt_conversions",
    # Receiving
    "rec": "receptions", "rec_yd": "receiving_yards", "rec_td": "receiving_tds",
    "rec_2pt": "receiving_2pt_conversions",
    # Fumbles
    "fum_lost": "fumbles_lost",
    # IDP (dst_* canonicals double for IDP scoring — see stat_ids.py)
    "idp_tkl": "def_tackles_total", "idp_tkl_solo": "def_tackles_solo",
    "idp_tkl_ast": "def_tackle_assists", "idp_sack": "dst_sacks",
    "idp_int": "dst_interceptions", "idp_ff": "def_fumbles_forced",
    "idp_fum_rec": "dst_fumbles_recovered", "idp_safe": "dst_safeties",
    # Kicking
    "xpm": "xp_made", "xpmiss": "xp_missed",
}
# Sleeper ADP keys carried through verbatim (draft-market context).
_SLEEPER_ADP_KEYS = ("adp_half_ppr", "adp_ppr", "adp_std", "adp_idp")


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


def sleeper_season_projections(season: int, refresh: bool = False) -> pd.DataFrame:
    """Sleeper's season-long per-stat projections as a canonical stat frame.

    Sleeper is our only source of real IDP projections (``idp_tkl``, ``idp_sack``,
    …) and also carries offense/kicking plus draft ADP. Each stat key is translated
    into our canonical stat columns (via :data:`_SLEEPER_STAT_MAP`) so the league
    :class:`~fantasy.valuation.scoring.ScoringEngine` scores it directly.

    Columns: ``sleeper_id, player_id (gsis|None), name, position, team`` + one
    column per projected canonical stat + the Sleeper ADP columns. Cached to
    ``sleeper_season_{season}.parquet``. Returns an empty frame on fetch failure.
    """
    path = settings.cache_dir / f"sleeper_season_{season}.parquet"
    # TTL'd cache: preseason projections keep moving. A stale cache is still the
    # fallback when the refetch fails, and an empty result is never cached (it
    # would mask the moment Sleeper publishes the season).
    cache_fresh = (path.exists()
                   and (time.time() - path.stat().st_mtime) < _SEASON_PROJ_TTL_SECONDS)
    if not refresh and cache_fresh:
        return pd.read_parquet(path)

    try:
        r = requests.get(_SLEEPER_SEASON.format(season=season),
                         headers={"User-Agent": "fantasy-app/0.1"}, timeout=30)
        r.raise_for_status()
        items = r.json()
    except Exception as e:  # noqa: BLE001
        if path.exists():
            log.warning("Sleeper season projections failed (%s); serving stale cache.", e)
            return pd.read_parquet(path)
        log.warning("Sleeper season projections failed (%s); returning empty.", e)
        return _empty_sleeper_season()

    xw = crosswalk()
    rows: list[dict] = []
    for it in items:
        stats = it.get("stats") or {}
        player = it.get("player") or {}
        sid = str(it.get("player_id"))
        first, last = player.get("first_name", ""), player.get("last_name", "")
        row: dict = {
            "sleeper_id": sid,
            "player_id": xw.from_sleeper(sid),
            "name": f"{first} {last}".strip() or None,
            "position": player.get("position"),
            "team": player.get("team") or player.get("team_abbr"),
        }
        for key, value in stats.items():
            canonical = _SLEEPER_STAT_MAP.get(key)
            if canonical is not None and value is not None:
                row[canonical] = row.get(canonical, 0.0) + float(value)
        for adp_key in _SLEEPER_ADP_KEYS:
            if stats.get(adp_key) is not None:
                row[adp_key] = float(stats[adp_key])
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        if path.exists():
            log.info("Sleeper season %s returned no rows; serving stale cache.", season)
            return pd.read_parquet(path)
        return _empty_sleeper_season()
    matched = df["player_id"].notna().sum() if "player_id" in df.columns else 0
    log.info("Sleeper season %s: %d players (%d gsis-matched).", season, len(df), matched)
    df.to_parquet(path, index=False)
    return df


def _empty_sleeper_season() -> pd.DataFrame:
    cols = ["sleeper_id", "player_id", "name", "position", "team"]
    return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})
