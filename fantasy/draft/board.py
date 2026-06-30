"""Draft value board — VOR over a no-future-leak preseason projection.

Preseason projection = each player's PRIOR-season league-scored points (a real,
leak-free estimate). Rookies / players with no prior season get an ADP-implied
projection (interpolated from where comparable players are drafted), so the board
covers the full draftable pool. VOR (cross-positional value) is computed on these
projections; ADP + sd are merged for the opponent/survival model.

For the LIVE draft you'd swap the projection for ESPN's preseason numbers; this
prior-season basis is what keeps self-play honest (no knowledge of the outcome).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from fantasy.data.nfl import season_totals
from fantasy.draft.adp import load_ffc_adp
from fantasy.league_settings import LeagueSettings
from fantasy.valuation.scoring import ScoringEngine
from fantasy.valuation.vor import compute_vor

log = logging.getLogger(__name__)


def _prior_points(season: int, league: LeagueSettings) -> dict[str, float]:
    """gsis -> prior-season total league points (the leak-free projection base)."""
    try:
        tot = season_totals([season - 1], ScoringEngine(league))
        return dict(zip(tot["player_id"], tot["pts"]))
    except Exception as e:  # noqa: BLE001
        log.warning("prior-season points unavailable for %s: %s", season - 1, e)
        return {}


def _market_implied(df: pd.DataFrame, prior: pd.Series) -> pd.Series:
    """Per-position interpolation of prior-season points vs ADP — i.e. the points
    a player drafted at this slot has historically produced (a market projection)."""
    implied = pd.Series(np.nan, index=df.index)
    for _pos, grp in df.groupby("position"):
        known = grp[prior.loc[grp.index].notna()]
        if len(known) < 3:
            continue
        xs = known["adp"].to_numpy(float)
        ys = prior.loc[known.index].to_numpy(float)
        order = np.argsort(xs)
        implied.loc[grp.index] = np.interp(grp["adp"].to_numpy(float), xs[order], ys[order])
    return implied


def _make_projection(df: pd.DataFrame, season: int, league: LeagueSettings) -> pd.Series:
    """Blend a player's own prior-season points with the market (ADP-implied)
    points. Starting from the market keeps the agent competitive with ADP; the
    own-production term tilts toward players the market is under/over-valuing."""
    prior = df["player_id"].map(_prior_points(season, league))
    implied = _market_implied(df, prior)
    proj = np.where(prior.notna(),
                    0.5 * prior.fillna(0.0) + 0.5 * implied.fillna(prior).fillna(0.0),
                    implied)
    return pd.Series(proj, index=df.index).fillna(0.0)


def build_replay_board(picks: pd.DataFrame, season: int, league: LeagueSettings) -> pd.DataFrame:
    """Draft board for replaying a REAL draft, using that draft's own order as ADP.

    ``picks`` has columns player_id, player_display_name, position, adp (the actual
    overall pick number). Used when external ADP for the season is unavailable;
    the projection is still the leak-free prior-season points.
    """
    df = picks.dropna(subset=["player_id"]).drop_duplicates("player_id").copy()
    df["proj"] = _make_projection(df, season, league)
    df["sd"] = (league.team_count / 2.0)  # modest spread around the realized slot
    return compute_vor(df[["player_id", "player_display_name", "position", "proj", "adp", "sd"]],
                       league).sort_values("adp").reset_index(drop=True)


def build_board(season: int, league: LeagueSettings, teams: int = 12,
                fmt: str = "ppr") -> pd.DataFrame:
    """Draft board for ``season``: player_id, name, position, proj, vor, adp, sd."""
    adp_df = load_ffc_adp(season, teams=teams, fmt=fmt).dropna(subset=["player_id"]).copy()
    adp_df = adp_df.drop_duplicates("player_id").reset_index(drop=True)
    adp_df["proj"] = _make_projection(adp_df, season, league)

    board = compute_vor(
        adp_df[["player_id", "name", "position", "proj", "adp", "sd"]].rename(
            columns={"name": "player_display_name"}),
        league,
    )
    return board.sort_values("adp").reset_index(drop=True)
