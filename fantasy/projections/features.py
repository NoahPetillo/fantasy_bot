"""Point-in-time feature engineering for weekly projections.

The cardinal rule: a feature for (player, season, week=t) may ONLY use data from
weeks strictly before t. Every trailing/expanding aggregate is shifted by one
game within the player's timeline, so the backtest can't peek at the outcome it's
predicting. Features are built on the LEAGUE-scored target (via a ScoringEngine),
so the whole pipeline is league-adaptive: change the league, re-score, retrain.

Feature families:
- form:        multi-window trailing (2/4/8) + exponentially-weighted + season-to-date
               mean/std of league fantasy points
- usage:       trailing volume + opportunity (targets, carries, target_share,
               air_yards_share, wopr) — opportunity is the sticky, predictive part
- efficiency:  trailing EPA / first downs / YAC / CPOE (how good the usage was)
- recency:     last-game values of key stats
- matchup:     trailing fantasy points the upcoming opponent allows to the position,
               plus its z-score across defenses (schedule-strength normalized)
- team:        trailing team plays / pass rate / target volume (pace & role context)
- regime:      new-team marker, games with current team, depth-chart usage rank
- context:     home/away, Vegas lines, weather, games played so far
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Columns we build trailing/expanding form features from (must exist in nflverse weekly).
_USAGE_COLS = [
    "targets", "receptions", "receiving_yards", "receiving_air_yards",
    "target_share", "air_yards_share", "wopr",
    "carries", "rushing_yards",
    "attempts", "completions", "passing_yards", "passing_tds",
]
# Efficiency columns: trailing-only (no *_last — single-game efficiency is noise).
_EFF_COLS = [
    "passing_epa", "rushing_epa", "receiving_epa", "passing_cpoe",
    "passing_first_downs", "rushing_first_downs", "receiving_first_downs",
    "receiving_yards_after_catch",
]
_FORM_COL = "y"  # league-scored fantasy points (added here)
_TRAIL = 4  # primary trailing window in games
_TRAIL_WINDOWS = (2, 8)  # extra windows so the model can learn position-specific recency


def build_features(
    weekly: pd.DataFrame, scoring_engine, trail: int = _TRAIL, context: bool = True
) -> pd.DataFrame:
    """Return a per-player-week feature table with a leak-free target column ``y``.

    Rows are kept only where a target exists (the player played week t). Week-1
    of a player's first observed season has no trailing history, so its trailing
    features are NaN — XGBoost handles NaN natively, so we keep those rows.

    When ``context`` is set, merges Vegas implied totals / spread / weather
    (current-week, known pre-game) and lagged offensive snap share.
    """
    df = weekly.copy()
    df = df[df["position"].isin(["QB", "RB", "WR", "TE"])].copy()
    df["y"] = scoring_engine.score_dataframe(df)
    # Ensure all usage/efficiency columns exist.
    for c in _USAGE_COLS + _EFF_COLS:
        if c not in df.columns:
            df[c] = 0.0
    df[_USAGE_COLS] = df[_USAGE_COLS].fillna(0.0)
    df[_EFF_COLS] = df[_EFF_COLS].fillna(0.0)
    # Total TDs scored/thrown — trailing rate is the TD-regression signal.
    td_parts = [c for c in ("passing_tds", "rushing_tds", "receiving_tds") if c in df.columns]
    df["tds_total"] = df[td_parts].fillna(0.0).sum(axis=1)

    seasons = sorted(df["season"].unique().tolist())
    if context:
        from fantasy.data.nfl import player_snap_share

        snaps = player_snap_share(seasons)
        if not snaps.empty:
            df = df.merge(snaps, on=["player_id", "season", "week"], how="left")
        else:
            df["offense_pct"] = float("nan")

    df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    g = df.groupby("player_id", sort=False)

    feat = pd.DataFrame(index=df.index)
    feat["player_id"] = df["player_id"]
    feat["player_display_name"] = df["player_display_name"]
    feat["position"] = df["position"]
    feat["team"] = df["team"]
    feat["opponent_team"] = df["opponent_team"]
    feat["season"] = df["season"]
    feat["week"] = df["week"]
    feat["y"] = df["y"]

    # games played so far this *career-in-data* (shifted: excludes current game)
    feat["games_so_far"] = g.cumcount()

    # form: trailing & expanding mean/std of league points (shifted by 1)
    shifted_y = g["y"].shift(1)
    sy = shifted_y.groupby(df["player_id"])
    feat["pts_trail_mean"] = sy.transform(lambda s: s.rolling(trail, min_periods=1).mean())
    feat["pts_trail_std"] = sy.transform(lambda s: s.rolling(trail, min_periods=2).std())
    feat["pts_season_mean"] = sy.transform(lambda s: s.expanding(min_periods=1).mean())
    feat["pts_last"] = shifted_y
    # multi-window + exponentially-weighted form — the model learns per-position
    # recency instead of being locked to a single window
    for w in _TRAIL_WINDOWS:
        feat[f"pts_trail{w}_mean"] = sy.transform(lambda s, w=w: s.rolling(w, min_periods=1).mean())
    feat["pts_ewm"] = sy.transform(lambda s: s.ewm(halflife=2.0, min_periods=1).mean())

    # usage: trailing means (shifted) + last-game value
    usage_cols = _USAGE_COLS + (["offense_pct"] if "offense_pct" in df.columns else [])
    for c in usage_cols:
        sh = g[c].shift(1)
        feat[f"{c}_trail"] = sh.groupby(df["player_id"]).transform(
            lambda s: s.rolling(trail, min_periods=1).mean()
        )
        feat[f"{c}_last"] = sh

    # efficiency + TD rate: trailing only (single-game efficiency is noise)
    for c in _EFF_COLS + ["tds_total"]:
        sh = g[c].shift(1)
        feat[f"{c}_trail"] = sh.groupby(df["player_id"]).transform(
            lambda s: s.rolling(trail, min_periods=1).mean()
        )

    # regime markers: joined a new team, and tenure with the current team —
    # lets the model react faster after trades/signings than pure trailing form
    prev_team = g["team"].shift(1)
    new_team = df["team"] != prev_team
    feat["new_team"] = new_team.astype(float)
    stint = new_team.groupby(df["player_id"]).cumsum()
    feat["games_with_team"] = df.groupby([df["player_id"], stint]).cumcount().astype(float)

    # depth-chart proxy: rank by trailing opportunity within (team, position)
    opp_trail = feat["targets_trail"].fillna(0.0) + feat["carries_trail"].fillna(0.0)
    feat["pos_usage_rank"] = opp_trail.groupby(
        [feat["season"], feat["week"], feat["team"], feat["position"]]
    ).rank(ascending=False, method="min")

    # team environment: trailing pace / pass rate / target volume (leak-free)
    feat = _add_team_volume(df, feat, trail)

    # matchup: trailing points the opponent's defense allows to this position
    feat = _add_defense_allowed(df, feat, trail)

    # game environment: Vegas implied total / spread / weather (current-week, legit)
    if context:
        from fantasy.data.nfl import team_week_context

        ctx = team_week_context(seasons)
        ctx_cols = ["season", "week", "team", "implied_total", "team_spread",
                    "game_total", "is_home", "temp", "wind", "is_outdoor"]
        feat = feat.merge(ctx[ctx_cols], on=["season", "week", "team"], how="left")
    else:
        feat["is_home"] = np.nan

    return feat


def future_frame(weekly: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """Synthetic unplayed rows for (season, week) so the week can be projected.

    Feature building only emits rows where a stat line exists, so before kickoff
    a week would produce an empty board — exactly the week a live league needs.
    And mid-week (Thursday stats published, Sun/Mon not yet played) the board
    would silently shrink to only the finished games. This synthesizes one row
    per recently-active player on each team that is SCHEDULED this week but has
    no stat rows yet, with the right opponent from the schedule and every stat
    NaN; all trailing features shift from prior games (strictly before this
    week), so the projection stays point-in-time. Fully-played weeks synthesize
    nothing, keeping retrospective backtests/audits unchanged.

    Players whose team is on bye that week get no row — they can't score, so they
    drop out of the board and the lineup optimizer can't start them.
    """
    from fantasy.data.nfl import week_opponents

    opponents = week_opponents(season, week)
    if not opponents:
        return pd.DataFrame()
    played_teams = set(
        weekly.loc[(weekly["season"] == season) & (weekly["week"] == week), "team"].unique()
    )
    missing_teams = set(opponents) - played_teams
    if not missing_teams:
        return pd.DataFrame()

    # Point-in-time candidate pool: this season's earlier weeks + last season.
    recent = weekly[
        (weekly["season"] == season - 1)
        | ((weekly["season"] == season) & (weekly["week"] < week))
    ]
    recent = recent[recent["position"].isin(["QB", "RB", "WR", "TE"])]
    if recent.empty:
        return pd.DataFrame()

    last = (
        recent.sort_values(["season", "week"])
        .groupby("player_id", as_index=False)
        .tail(1)[["player_id", "player_display_name", "position", "team"]]
    )
    last = last[last["team"].isin(missing_teams)]
    last["opponent_team"] = last["team"].map(opponents)
    fut = pd.DataFrame(index=range(len(last)), columns=weekly.columns)
    for c in ("player_id", "player_display_name", "position", "team", "opponent_team"):
        fut[c] = last[c].to_numpy()
    fut["season"] = season
    fut["week"] = week
    if "season_type" in fut.columns:
        fut["season_type"] = "REG"
    # Stat columns stay NaN: the row exists to receive trailing features, not stats.
    return fut


def _add_team_volume(df: pd.DataFrame, feat: pd.DataFrame, trail: int) -> pd.DataFrame:
    """Trailing team plays / pass rate / target volume merged per (season, week, team).

    A player's opportunity is capped by his offense's volume; a role on a fast,
    pass-heavy team is worth more than the same role on a slow one. All values are
    shifted within the team's timeline, so week t only sees prior games.
    """
    team_wk = (
        df.groupby(["season", "week", "team"], as_index=False)
        .agg(team_targets=("targets", "sum"), team_carries=("carries", "sum"),
             team_attempts=("attempts", "sum"))
        .sort_values(["team", "season", "week"])
    )
    team_wk["team_plays"] = team_wk["team_attempts"] + team_wk["team_carries"]
    team_wk["team_pass_rate"] = team_wk["team_attempts"] / team_wk["team_plays"].replace(0, np.nan)
    tg = team_wk.groupby("team")
    for c in ("team_plays", "team_pass_rate", "team_targets"):
        team_wk[f"{c}_trail"] = tg[c].transform(
            lambda s: s.shift(1).rolling(trail, min_periods=1).mean()
        )
    merged = feat.merge(
        team_wk[["season", "week", "team",
                 "team_plays_trail", "team_pass_rate_trail", "team_targets_trail"]],
        on=["season", "week", "team"], how="left",
    )
    merged.index = feat.index
    return merged


def _add_defense_allowed(df: pd.DataFrame, feat: pd.DataFrame, trail: int) -> pd.DataFrame:
    """Trailing fantasy points each defense allows to each position (leak-free)."""
    # Points a defense (opponent_team) allowed to a position in a given week.
    allowed = (
        df.groupby(["season", "week", "opponent_team", "position"], as_index=False)["y"]
        .sum()
        .rename(columns={"opponent_team": "def_team", "y": "pts_allowed"})
        .sort_values(["def_team", "position", "season", "week"])
    )
    # Trailing mean of points allowed, shifted so week t isn't included.
    allowed["def_allow_trail"] = (
        allowed.groupby(["def_team", "position"])["pts_allowed"]
        .apply(lambda s: s.shift(1).rolling(trail, min_periods=1).mean())
        .reset_index(level=[0, 1], drop=True)
    )
    # Z-score across defenses that week: "how soft is this matchup" in units
    # comparable across seasons and scoring environments.
    grp = allowed.groupby(["season", "week", "position"])["def_allow_trail"]
    sd = grp.transform("std").replace(0, np.nan)
    allowed["def_allow_z"] = (allowed["def_allow_trail"] - grp.transform("mean")) / sd
    merged = feat.merge(
        allowed[["season", "week", "def_team", "position", "def_allow_trail", "def_allow_z"]],
        left_on=["season", "week", "opponent_team", "position"],
        right_on=["season", "week", "def_team", "position"],
        how="left",
    )
    merged = merged.drop(columns=["def_team"])
    merged.index = feat.index
    return merged


def feature_columns(feat: pd.DataFrame) -> list[str]:
    """The numeric model inputs (everything except identifiers and target)."""
    drop = {
        "player_id", "player_display_name", "position", "team",
        "opponent_team", "season", "week", "y",
    }
    return [c for c in feat.columns if c not in drop]
