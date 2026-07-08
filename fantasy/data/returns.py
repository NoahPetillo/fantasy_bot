"""Kick/punt return valuation — the format's biggest edge.

In leagues that score individual return yards (0.25/yd here), a full-time kick
returner is worth ~350-500 pts/season, yet every consensus projection carries
ZERO return yardage. This module supplies the missing layer:

- :func:`player_return_stats` — historical per-player return production (from the
  weekly frame, which already carries return columns; no play-by-play needed).
- :func:`current_returners` — who currently holds the KR/PR job (from depth charts).
- :func:`return_points_overlay` — gsis -> expected SEASON return points under the
  league's scoring, blending each holder's own prior production with the
  league-wide full-time-returner median and discounting for job-security risk.

The era anchor is the most recent completed season (post-2025 dynamic-kickoff
rules ≈ 78% return rate), so the full-time-returner median reflects today's game.
"""

from __future__ import annotations

import logging

import pandas as pd

from fantasy.config import settings
from fantasy.data.nfl import load_depth_charts, load_weekly
from fantasy.league_settings import LeagueSettings

log = logging.getLogger(__name__)

# The canonical scoring stats this overlay owns EXCLUSIVELY. Consumers (the
# season board) must exclude these from per-stat row scoring — the overlay is
# the single source of return-yardage points, so a projection source that
# starts publishing them can never double-count.
OVERLAY_OWNED_STATS: frozenset[str] = frozenset(
    {"kickoff_return_yards", "punt_return_yards"}
)

# Depth-chart roles that mean "kick returner" / "punt returner".
_KR_ROLES = {"KR", "KOR"}
_PR_ROLES = {"PR"}
# A "full-time" returner threshold (return-count over a season) used to compute the
# league-wide median per-game yardage — filters out one-off / emergency returners.
_FULLTIME_KR_MIN = 20  # ~20+ kickoff returns ≈ a real return job in the modern era
_FULLTIME_PR_MIN = 15  # punt returns are rarer per game than kickoffs
# Weight on the returner's OWN prior per-game yardage vs the league median (the
# rest). New/unproven holders lean on the league median; proven ones on their own.
_OWN_YARDS_WEIGHT = 0.5
# The #2 depth-chart returner gets a fraction of a full-time expectation (spot
# duty / injury insurance), and the era's full season is 17 games.
_RANK2_SHARE = 0.25
_SEASON_GAMES = 17
# Season used as the "modern era" anchor for the full-time-returner medians when a
# more specific completed season isn't obvious.
_ERA_SEASON = 2025


def player_return_stats(seasons: list[int], refresh: bool = False) -> pd.DataFrame:
    """Per (player_id, season) kick/punt return production from the weekly frame.

    Columns: ``player_id, player_display_name, season, kr_yards, pr_yards,
    kr_count, pr_count, games``. The weekly frame already carries return columns
    for every player, so this is a straight groupby.
    """
    key = f"return_stats_{min(seasons)}_{max(seasons)}"
    path = settings.cache_dir / f"{key}.parquet"
    if not refresh and path.exists():
        return pd.read_parquet(path)

    df = load_weekly(seasons, refresh=refresh)
    for col in ("kickoff_return_yards", "punt_return_yards",
                "kickoff_returns", "punt_returns"):
        if col not in df.columns:
            df[col] = 0.0
    grp = (
        df.groupby(["player_id", "player_display_name", "season"], dropna=False)
        .agg(
            kr_yards=("kickoff_return_yards", "sum"),
            pr_yards=("punt_return_yards", "sum"),
            kr_count=("kickoff_returns", "sum"),
            pr_count=("punt_returns", "sum"),
            games=("week", "nunique"),
        )
        .reset_index()
    )
    grp.to_parquet(path, index=False)
    return grp


def current_returners(season: int, refresh: bool = False) -> pd.DataFrame:
    """Who holds the KR/PR job now, from the latest available depth-chart week.

    Rows: ``gsis_id, name, team, role ('KR'|'PR'), rank (1|2)``. 'KOR' folds into
    'KR'. If depth charts for ``season`` aren't published yet (e.g. July), falls
    back to the latest prior season with data and marks ``stale_season`` True.
    """
    used_season = season
    stale = False
    depth = None
    for s in (season, season - 1, season - 2):
        try:
            d = load_depth_charts(s, refresh=refresh and s == season)
        except Exception as e:  # noqa: BLE001
            log.info("Depth charts for %s unavailable (%s).", s, e)
            continue
        if d is not None and not d.empty:
            depth, used_season, stale = d, s, (s != season)
            break
    if depth is None or depth.empty:
        log.warning("No depth charts available near season %s.", season)
        return pd.DataFrame(
            columns=["gsis_id", "name", "team", "role", "rank", "stale_season"]
        )

    out = _parse_returner_roles(depth)
    out["stale_season"] = stale
    if stale:
        log.info("current_returners: %s unavailable; using %s depth charts.",
                 season, used_season)
    return out


def _parse_returner_roles(depth: pd.DataFrame) -> pd.DataFrame:
    """Extract KR/PR holders from a depth-chart frame, tolerant of both the
    current nflreadpy schema (``pos_abb``/``pos_rank``/``player_name``/``dt``) and
    the legacy schema (``depth_position``/``depth_team``/``full_name``/``week``)."""
    cols = set(depth.columns)
    if "pos_abb" in cols:  # current nflreadpy schema
        role_c, rank_c, name_c, team_c = "pos_abb", "pos_rank", "player_name", "team"
        recency_c = "dt" if "dt" in cols else None
    else:  # legacy cached schema
        role_c, rank_c, name_c, team_c = ("depth_position", "depth_team",
                                          "full_name", "club_code")
        recency_c = "week" if "week" in cols else None

    ret = depth[depth[role_c].isin(_KR_ROLES | _PR_ROLES)].copy()
    # Roles change over time; keep only the most recent snapshot/week.
    if recency_c and ret[recency_c].notna().any():
        ret = ret[ret[recency_c] == ret[recency_c].max()]
    ret["role"] = ret[role_c].apply(lambda r: "PR" if r in _PR_ROLES else "KR")
    ret["rank"] = pd.to_numeric(ret[rank_c], errors="coerce").fillna(1).astype(int)
    out = pd.DataFrame({
        "gsis_id": ret.get("gsis_id"),
        "name": ret.get(name_c),
        "team": ret.get(team_c),
        "role": ret["role"],
        "rank": ret["rank"],
    }).dropna(subset=["gsis_id"])
    # Keep the best (lowest-rank) listing per player+role.
    out = out.sort_values("rank").drop_duplicates(["gsis_id", "role"]).reset_index(drop=True)
    return out


def _fulltime_medians(hist: pd.DataFrame) -> tuple[float, float]:
    """League-wide median per-game return yards for a *full-time* KR and PR,
    computed from the era-anchor season (falls back to all history if absent)."""
    era = hist[hist["season"] == _ERA_SEASON]
    if era.empty:
        era = hist
    kr = era[(era["kr_count"] >= _FULLTIME_KR_MIN) & (era["games"] > 0)]
    pr = era[(era["pr_count"] >= _FULLTIME_PR_MIN) & (era["games"] > 0)]
    kr_med = float((kr["kr_yards"] / kr["games"]).median()) if len(kr) else 0.0
    pr_med = float((pr["pr_yards"] / pr["games"]).median()) if len(pr) else 0.0
    return kr_med, pr_med


def _own_per_game(hist: pd.DataFrame, gsis_id: str) -> tuple[float | None, float | None]:
    """A holder's OWN most-recent per-game KR/PR yards (None if no prior returns)."""
    rows = hist[hist["player_id"] == gsis_id]
    if rows.empty:
        return None, None
    latest = rows.sort_values("season").iloc[-1]
    g = latest["games"] or 0
    kr = (latest["kr_yards"] / g) if g and latest["kr_count"] else None
    pr = (latest["pr_yards"] / g) if g and latest["pr_count"] else None
    return kr, pr


def return_points_overlay(
    league: LeagueSettings, season: int, discount: float = 0.20
) -> dict[str, float]:
    """gsis_id -> expected SEASON return points under ``league`` scoring.

    Model
    -----
    Returns ``{}`` unless the league scores return yards (``kickoff_return_yards``
    or ``punt_return_yards``). For each current depth-chart returner:

    * expected per-game yards blend the holder's OWN prior per-game production
      (weight :data:`_OWN_YARDS_WEIGHT`) with the league-wide full-time-returner
      median (the remainder). Unproven holders lean fully on the median.
    * that per-game figure scales to a full season (× :data:`_SEASON_GAMES`),
      then × the league's per-yard points, then × ``(1 - discount)`` for
      job-security risk. Depth-rank-2 holders get :data:`_RANK2_SHARE` of a
      full-time expectation.

    KR and PR contributions are summed per player (a player can hold both jobs).
    """
    kr_pts = float(league.scoring.get("kickoff_return_yards", 0.0) or 0.0)
    pr_pts = float(league.scoring.get("punt_return_yards", 0.0) or 0.0)
    if not kr_pts and not pr_pts:
        return {}

    hist = player_return_stats([_ERA_SEASON - 1, _ERA_SEASON])
    kr_med, pr_med = _fulltime_medians(hist)
    holders = current_returners(season)
    if holders.empty:
        return {}

    keep = 1.0 - discount
    overlay: dict[str, float] = {}
    for r in holders.itertuples(index=False):
        role_pts = kr_pts if r.role == "KR" else pr_pts
        if not role_pts:
            continue
        median = kr_med if r.role == "KR" else pr_med
        own_kr, own_pr = _own_per_game(hist, r.gsis_id)
        own = own_kr if r.role == "KR" else own_pr
        if own is not None:
            per_game = _OWN_YARDS_WEIGHT * own + (1.0 - _OWN_YARDS_WEIGHT) * median
        else:
            per_game = median
        share = 1.0 if r.rank <= 1 else _RANK2_SHARE
        pts = per_game * _SEASON_GAMES * role_pts * keep * share
        overlay[r.gsis_id] = overlay.get(r.gsis_id, 0.0) + round(pts, 2)
    return overlay
