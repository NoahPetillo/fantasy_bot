"""nflverse data spine (via nflreadpy), with local parquet caching.

Provides the historical ground truth for training and backtesting:
- weekly player stats (incl. opportunity metrics: target_share, air_yards_share, wopr)
- the ff player-id crosswalk (espn_id <-> sleeper_id <-> gsis_id <-> ...)

Everything returns pandas (downstream modeling/scoring uses pandas). Results are
cached to ``data/cache/*.parquet`` so repeated runs are offline and fast.
"""

from __future__ import annotations

import logging

import pandas as pd

from fantasy.config import settings

log = logging.getLogger(__name__)

# nflverse splits fumbles-lost across phases; fantasy scoring wants the total.
_FUMBLE_LOST_PARTS = ["sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost"]


def _cache_path(name: str) -> str:
    return str(settings.cache_dir / f"{name}.parquet")


def load_weekly(seasons: list[int], refresh: bool = False) -> pd.DataFrame:
    """Weekly player stats for the given seasons (regular + post), normalized.

    Adds a synthesized ``fumbles_lost`` column (sum of phase fumbles lost) so the
    league ScoringEngine can score it directly.
    """
    key = f"weekly_{min(seasons)}_{max(seasons)}"
    path = _cache_path(key)
    if not refresh:
        try:
            df = pd.read_parquet(path)
            log.info("Loaded weekly stats from cache: %s (%d rows)", path, len(df))
            # Re-normalize on cache hits too: caches written before a
            # normalization rule existed (e.g. the synthesized IDP columns)
            # would otherwise silently miss it forever. Idempotent.
            return _normalize_weekly(df)
        except (FileNotFoundError, OSError):
            pass

    import nflreadpy as nfl

    log.info("Downloading weekly stats for seasons %s ...", seasons)
    try:
        raw = nfl.load_player_stats(seasons=seasons).to_pandas()
        loaded = list(seasons)
    except Exception as e:  # noqa: BLE001
        # A season with no published data yet (e.g. the current season before its
        # first games) 404s and fails the whole batch. Fall back to per-season so
        # the seasons that ARE out still load; skip the ones that aren't published.
        log.warning("Batch weekly load failed (%s); loading season-by-season.", e)
        frames, loaded = [], []
        for s in seasons:
            try:
                frames.append(nfl.load_player_stats(seasons=[s]).to_pandas())
                loaded.append(s)
            except Exception as se:  # noqa: BLE001
                log.warning("Weekly stats for %s unavailable (skipping): %s", s, se)
        if not frames:
            raise
        raw = pd.concat(frames, ignore_index=True)
    df = _normalize_weekly(raw)
    # Only cache when every requested season loaded — a partial result would poison
    # the range-keyed cache and hide a season's data once it's finally published.
    if set(loaded) == set(seasons):
        df.to_parquet(path, index=False)
        log.info("Cached weekly stats -> %s (%d rows)", path, len(df))
    else:
        log.info("Loaded weekly stats for %s (partial; not cached)", loaded)
    return df


# Granular nflverse defensive labels -> ESPN-style IDP positions (plus D/ST alias).
_POSITION_NORMALIZE = {
    "DEF": "D/ST", "DST": "D/ST",
    "ILB": "LB", "MLB": "LB", "OLB": "LB",
    "FS": "S", "SS": "S", "SAF": "S",
    "NT": "DT", "EDGE": "DE",
}


def _normalize_weekly(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    present = [c for c in _FUMBLE_LOST_PARTS if c in df.columns]
    df["fumbles_lost"] = df[present].fillna(0).sum(axis=1) if present else 0.0
    # Normalize position labels to our canonical tokens if present.
    if "position" in df.columns:
        df["position"] = df["position"].replace(_POSITION_NORMALIZE)
    # IDP conveniences: ESPN scores total tackles / half-sacks; nflverse splits
    # them. Only synthesized when absent, so a source-provided column wins.
    if "def_tackles_total" not in df.columns and "def_tackles_solo" in df.columns:
        assists = (
            df["def_tackle_assists"].fillna(0)
            if "def_tackle_assists" in df.columns else 0.0
        )
        df["def_tackles_total"] = df["def_tackles_solo"].fillna(0) + assists
    if "def_half_sacks" not in df.columns and "def_sacks" in df.columns:
        df["def_half_sacks"] = df["def_sacks"].fillna(0) * 2.0
    return df


def team_week_context(seasons: list[int], refresh: bool = False) -> pd.DataFrame:
    """Per (season, week, team) game environment from schedules + betting lines.

    Vegas lines are known BEFORE kickoff, so these are legitimate point-in-time
    features (no lag needed): implied team total, team spread, game total, and
    outdoor weather. One row per team per game (home + away exploded).
    """
    key = f"context_{min(seasons)}_{max(seasons)}"
    path = _cache_path(key)
    if not refresh:
        try:
            return pd.read_parquet(path)
        except (FileNotFoundError, OSError):
            pass
    import nflreadpy as nfl

    sch = nfl.load_schedules(seasons=seasons).to_pandas()
    sch = sch[sch["total_line"].notna() | sch["spread_line"].notna()].copy()
    outdoor = ~sch["roof"].isin(["dome", "closed"])
    base = dict(
        season=sch["season"], week=sch["week"], game_total=sch["total_line"],
        temp=sch["temp"].where(outdoor), wind=sch["wind"].where(outdoor),
        is_outdoor=outdoor.astype(float),
    )
    home = pd.DataFrame({
        **base, "team": sch["home_team"], "opponent_team": sch["away_team"], "is_home": 1.0,
        "team_spread": sch["spread_line"],
        "implied_total": (sch["total_line"] + sch["spread_line"]) / 2.0,
    })
    away = pd.DataFrame({
        **base, "team": sch["away_team"], "opponent_team": sch["home_team"], "is_home": 0.0,
        "team_spread": -sch["spread_line"],
        "implied_total": (sch["total_line"] - sch["spread_line"]) / 2.0,
    })
    ctx = pd.concat([home, away], ignore_index=True)
    ctx.to_parquet(path, index=False)
    log.info("Cached team-week context -> %s (%d rows)", path, len(ctx))
    return ctx


def season_schedule(season: int, refresh: bool = False) -> pd.DataFrame:
    """The raw regular-season schedule (one row per game), cached.

    Unlike ``team_week_context`` this keeps games with no betting lines yet, so it
    can answer "who plays whom in week W" for FUTURE weeks — the projection
    service needs that to build a board for a week that hasn't been played.
    """
    key = f"schedule_{season}"
    path = _cache_path(key)
    if not refresh:
        try:
            return pd.read_parquet(path)
        except (FileNotFoundError, OSError):
            pass
    import nflreadpy as nfl

    sch = nfl.load_schedules(seasons=[season]).to_pandas()
    if "game_type" in sch.columns:
        sch = sch[sch["game_type"] == "REG"]
    out = sch[["season", "week", "home_team", "away_team"]].copy()
    out.to_parquet(path, index=False)
    log.info("Cached schedule -> %s (%d games)", path, len(out))
    return out


def week_opponents(season: int, week: int) -> dict[str, str]:
    """team -> opponent for one week. Teams absent from the map are on bye."""
    sch = season_schedule(season)
    wk = sch[sch["week"] == week]
    opp: dict[str, str] = {}
    for r in wk.itertuples(index=False):
        opp[r.home_team] = r.away_team
        opp[r.away_team] = r.home_team
    return opp


def team_bye_weeks(season: int) -> dict[str, int]:
    """team -> bye week (the regular-season week the team doesn't play)."""
    sch = season_schedule(season)
    if sch.empty:
        return {}
    weeks = set(range(1, int(sch["week"].max()) + 1))
    plays: dict[str, set[int]] = {}
    for r in sch.itertuples(index=False):
        plays.setdefault(r.home_team, set()).add(int(r.week))
        plays.setdefault(r.away_team, set()).add(int(r.week))
    return {t: min(weeks - w) for t, w in plays.items() if weeks - w}


def player_snap_share(seasons: list[int], refresh: bool = False) -> pd.DataFrame:
    """Per (player_id, season, week) offensive snap share, keyed to gsis player_id.

    Snap counts use PFR ids; we map them to gsis ids via the ff crosswalk so they
    join to weekly stats. Snap share for week t is NOT known pre-game, so callers
    must lag it (handled in features).
    """
    key = f"snaps_{min(seasons)}_{max(seasons)}"
    path = _cache_path(key)
    if not refresh:
        try:
            return pd.read_parquet(path)
        except (FileNotFoundError, OSError):
            pass
    import nflreadpy as nfl

    snaps = nfl.load_snap_counts(seasons=seasons).to_pandas()
    xwalk = load_player_ids()
    cols = {c.lower(): c for c in xwalk.columns}
    pfr_c, gsis_c = cols.get("pfr_id"), cols.get("gsis_id")
    out_cols = ["season", "week", "player_id", "offense_pct"]
    if pfr_c and gsis_c and "pfr_player_id" in snaps.columns:
        m = (
            xwalk[[pfr_c, gsis_c]].dropna()
            .rename(columns={pfr_c: "pfr_player_id", gsis_c: "player_id"})
            .drop_duplicates("pfr_player_id")
        )
        snaps = snaps.merge(m, on="pfr_player_id", how="left")
        out = snaps.dropna(subset=["player_id"]).drop_duplicates(["player_id", "season", "week"])[out_cols]
    else:
        log.warning("Snap crosswalk unavailable; snap-share feature will be empty.")
        out = pd.DataFrame(columns=out_cols)
    out.to_parquet(path, index=False)
    log.info("Cached snap share -> %s (%d rows)", path, len(out))
    return out


def load_depth_charts(season: int, refresh: bool = False) -> pd.DataFrame:
    """Weekly depth charts for ``season`` (via nflreadpy), cached per season.

    Includes special-teams roles (``depth_position`` in {'KR','KOR','PR',...}) with
    ``depth_team`` as the string/numeric rank — the source for who returns kicks and
    punts. Columns pass through nflreadpy verbatim (club_code, gsis_id, full_name,
    depth_position, depth_team, formation, week, season, ...).
    """
    key = f"depth_{season}"
    path = _cache_path(key)
    if not refresh:
        try:
            return pd.read_parquet(path)
        except (FileNotFoundError, OSError):
            pass
    import nflreadpy as nfl

    df = nfl.load_depth_charts(seasons=[season]).to_pandas()
    df.to_parquet(path, index=False)
    log.info("Cached depth charts -> %s (%d rows)", path, len(df))
    return df


def load_player_ids(refresh: bool = False) -> pd.DataFrame:
    """The ff player-id crosswalk — the linchpin join table across data sources."""
    path = _cache_path("ff_playerids")
    if not refresh:
        try:
            return pd.read_parquet(path)
        except (FileNotFoundError, OSError):
            pass
    import nflreadpy as nfl

    log.info("Downloading ff player-id crosswalk ...")
    df = nfl.load_ff_playerids().to_pandas()
    df.to_parquet(path, index=False)
    log.info("Cached crosswalk -> %s (%d rows)", path, len(df))
    return df


def season_totals(seasons: list[int], scoring_engine, refresh: bool = False) -> pd.DataFrame:
    """Per-player season fantasy-point totals under a specific league ScoringEngine.

    Demonstrates league-adaptive scoring on real history: the engine encapsulates
    the league's exact rules, so this total is what that player would have scored
    in *this* league.
    """
    df = load_weekly(seasons, refresh=refresh)
    df = df.assign(fantasy_points_league=scoring_engine.score_dataframe(df))
    grp = (
        df.groupby(["player_id", "player_display_name", "position", "season"], dropna=False)
        .agg(
            games=("week", "nunique"),
            pts=("fantasy_points_league", "sum"),
            pts_nflverse_ppr=("fantasy_points_ppr", "sum"),
        )
        .reset_index()
    )
    grp["ppg"] = (grp["pts"] / grp["games"]).round(2)
    return grp.sort_values("pts", ascending=False)
