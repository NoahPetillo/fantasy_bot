"""Head-coach valuation — the pure-EV slot in coach-scoring leagues.

An HC slot scores off the team's game *result* (here +5 win / -5 loss), so a
coach's fantasy value is entirely their team's win probability. Two views:

- :func:`hc_draft_ev` — expected SEASON points per team, from win probabilities.
  Win probabilities come from the-odds-api season win totals when a key is
  configured (same key as the props source); otherwise from prior-season win%
  regressed halfway to .500 (a cheap, no-dependency, no-network estimate).
- :func:`hc_stream_ev` — the weekly-stream vs best-drafted-coach EV deltas that
  drive the "stream the biggest weekly favorite, don't draft-and-hold" guidance.

Research anchors the stream constants: the biggest weekly moneyline favorite wins
~78% of the time; the best preseason win-total team projects ~68% per game.
"""

from __future__ import annotations

import logging

from fantasy.config import settings
from fantasy.league_settings import LeagueSettings

log = logging.getLogger(__name__)

_SEASON_GAMES = 17
# Research-derived per-game win probabilities (see module docstring).
_STREAM_FAVORITE_WINP = 0.78  # typical biggest weekly moneyline favorite
_BEST_DRAFTED_WINP = 0.68  # best preseason win-total team, held all season
# Fallback win% is regressed this far toward .500 (small samples / coaching change).
_REGRESSION_TO_MEAN = 0.50
_ODDS_FUTURES_URL = (
    "https://api.the-odds-api.com/v4/sports/americanfootball_nfl_super_bowl_winner/odds"
)


def hc_expected_points(league: LeagueSettings, win_prob: float) -> float:
    """Expected HC points for one game at win probability ``win_prob``.

    ``p·win_pts + (1-p)·loss_pts``. Ties are ignored: they occur in ~0.4% of NFL
    games, so even a scored ``hc_team_tie`` rule moves weekly EV by well under
    0.05 pts — not worth complicating the win-probability model for.
    """
    p = min(max(float(win_prob), 0.0), 1.0)
    win_pts = float(league.scoring.get("hc_team_win", 0.0) or 0.0)
    loss_pts = float(league.scoring.get("hc_team_loss", 0.0) or 0.0)
    return round(p * win_pts + (1.0 - p) * loss_pts, 4)


def hc_draft_ev(league: LeagueSettings, season: int):
    """Per-NFL-team head-coach draft EV as a DataFrame.

    Columns: ``player_id ('HC:<team>'), team, coach_label ('HC <team>'),
    position ('HC'), win_prob, expected_season_points``. Win probabilities come
    from the-odds-api futures if a key is set, else from a prior-season fallback.
    """
    import pandas as pd

    win_probs = _win_probabilities(season)
    rows = []
    for team, p in sorted(win_probs.items()):
        ev = round(hc_expected_points(league, p) * _SEASON_GAMES, 2)
        rows.append({
            "player_id": f"HC:{team}",
            "team": team,
            "coach_label": f"HC {team}",
            "position": "HC",
            "win_prob": round(p, 4),
            "expected_season_points": ev,
        })
    return pd.DataFrame(rows)


def hc_stream_ev(league: LeagueSettings) -> dict:
    """Weekly + season EV for streaming the biggest favorite vs the best drafted HC.

    The stream figure assumes you can rostere the week's biggest moneyline favorite
    each week (~78% win prob); the drafted figure holds the best preseason
    win-total team (~68%). Season figures scale by :data:`_SEASON_GAMES`.
    """
    stream_wk = hc_expected_points(league, _STREAM_FAVORITE_WINP)
    drafted_wk = hc_expected_points(league, _BEST_DRAFTED_WINP)
    return {
        "stream_weekly_pts": round(stream_wk, 2),
        "best_drafted_weekly_pts": round(drafted_wk, 2),
        "stream_season_pts": round(stream_wk * _SEASON_GAMES, 2),
        "best_drafted_season_pts": round(drafted_wk * _SEASON_GAMES, 2),
    }


# ── win probabilities ─────────────────────────────────────────────────────────
def _win_probabilities(season: int) -> dict[str, float]:
    """team -> per-game win probability. the-odds-api futures if available, else
    a prior-season fallback."""
    if settings.odds_api_key:
        probs = _win_probs_from_odds()
        if probs:
            return probs
    return _win_probs_from_prior_season(season)


def _win_probs_from_odds() -> dict[str, float]:
    """Best-effort per-game win prob from de-vigged Super Bowl futures.

    Futures aren't a per-game probability, so we map the (normalized) title odds
    onto a plausible per-game win-rate band — this is a coarse ordering signal, and
    the whole path degrades to the prior-season fallback if the plan/endpoint can't
    serve futures.
    """
    import requests

    from fantasy.projections.props import american_to_prob
    try:
        r = requests.get(_ODDS_FUTURES_URL, params={
            "apiKey": settings.odds_api_key, "regions": "us",
            "markets": "outrights", "oddsFormat": "american"}, timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception as e:  # noqa: BLE001
        log.info("HC win-total futures unavailable (%s); using prior-season fallback.", e)
        return {}

    raw: dict[str, float] = {}
    for ev in data if isinstance(data, list) else [data]:
        for book in ev.get("bookmakers", []):
            for mkt in book.get("markets", []):
                for oc in mkt.get("outcomes", []):
                    name = oc.get("name")
                    price = oc.get("price")
                    if name and price is not None:
                        raw.setdefault(name, american_to_prob(float(price)))
    if not raw:
        return {}
    # Map title-implied strength onto a per-game win-rate band [.40, .70].
    lo, hi = min(raw.values()), max(raw.values())
    span = (hi - lo) or 1.0
    return {name: 0.40 + 0.30 * ((p - lo) / span) for name, p in raw.items()}


def _win_probs_from_prior_season(season: int) -> dict[str, float]:
    """team -> per-game win prob from the prior season's record, regressed to .500."""
    import nflreadpy as nfl

    try:
        sch = nfl.load_schedules(seasons=[season - 1]).to_pandas()
    except Exception as e:  # noqa: BLE001
        log.warning("Prior-season schedule for HC fallback unavailable (%s).", e)
        return {}
    if "game_type" in sch.columns:
        sch = sch[sch["game_type"] == "REG"]
    sch = sch[sch["home_score"].notna() & sch["away_score"].notna()]
    if sch.empty:
        log.info("No completed prior-season games for HC win%% fallback (%s).", season - 1)
        return {}

    wins: dict[str, float] = {}
    games: dict[str, int] = {}
    for r in sch.itertuples(index=False):
        margin = float(r.home_score) - float(r.away_score)
        for team, sign in ((r.home_team, 1.0), (r.away_team, -1.0)):
            games[team] = games.get(team, 0) + 1
            if margin * sign > 0:
                wins[team] = wins.get(team, 0.0) + 1.0
            elif margin == 0:
                wins[team] = wins.get(team, 0.0) + 0.5
    out: dict[str, float] = {}
    for team, g in games.items():
        raw = wins.get(team, 0.0) / g if g else 0.5
        out[team] = _REGRESSION_TO_MEAN * raw + (1.0 - _REGRESSION_TO_MEAN) * 0.5
    return out
