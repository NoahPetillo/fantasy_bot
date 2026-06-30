"""Point-in-time feature engineering for weekly projections.

The cardinal rule: a feature for (player, season, week=t) may ONLY use data from
weeks strictly before t. Every trailing/expanding aggregate is shifted by one
game within the player's timeline, so the backtest can't peek at the outcome it's
predicting. Features are built on the LEAGUE-scored target (via a ScoringEngine),
so the whole pipeline is league-adaptive: change the league, re-score, retrain.

Feature families:
- form:        trailing-N and season-to-date mean/std of league fantasy points
- usage:       trailing volume + opportunity (targets, carries, target_share,
               air_yards_share, wopr) — opportunity is the sticky, predictive part
- recency:     last-game values of key stats
- matchup:     trailing fantasy points the upcoming opponent allows to the position
- context:     home/away, games played so far
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
_FORM_COL = "y"  # league-scored fantasy points (added here)
_TRAIL = 4  # trailing window in games


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
    # Ensure all usage columns exist.
    for c in _USAGE_COLS:
        if c not in df.columns:
            df[c] = 0.0
    df[_USAGE_COLS] = df[_USAGE_COLS].fillna(0.0)

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
    feat["pts_trail_mean"] = shifted_y.groupby(df["player_id"]).transform(
        lambda s: s.rolling(trail, min_periods=1).mean()
    )
    feat["pts_trail_std"] = shifted_y.groupby(df["player_id"]).transform(
        lambda s: s.rolling(trail, min_periods=2).std()
    )
    feat["pts_season_mean"] = shifted_y.groupby(df["player_id"]).transform(
        lambda s: s.expanding(min_periods=1).mean()
    )
    feat["pts_last"] = shifted_y

    # usage: trailing means (shifted) + last-game value
    usage_cols = _USAGE_COLS + (["offense_pct"] if "offense_pct" in df.columns else [])
    for c in usage_cols:
        sh = g[c].shift(1)
        feat[f"{c}_trail"] = sh.groupby(df["player_id"]).transform(
            lambda s: s.rolling(trail, min_periods=1).mean()
        )
        feat[f"{c}_last"] = sh

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
    merged = feat.merge(
        allowed[["season", "week", "def_team", "position", "def_allow_trail"]],
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
