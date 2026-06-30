"""Player-prop projections — the sharpest single source (money-weighted markets).

Converts Vegas player-prop lines (pass/rush/rec yards, receptions, TD odds) into
projected fantasy points in the league's scoring, as a consensus source. Yards/
reception lines ARE the market's expected value; TD odds are de-vigged to an
expected-TD count. Requires ODDS_API_KEY (the-odds-api.com; free tier exists but
player props cost credits) — off and harmless without one, like the X adapter.
"""

from __future__ import annotations

import logging

from fantasy.config import settings
from fantasy.data.ids import crosswalk
from fantasy.league_settings import LeagueSettings
from fantasy.valuation.scoring import ScoringEngine

log = logging.getLogger(__name__)

API = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"
# the-odds-api market key -> our canonical stat (yards/receptions: line = expected value)
_LINE_MARKETS = {
    "player_pass_yds": "passing_yards", "player_rush_yds": "rushing_yards",
    "player_reception_yds": "receiving_yards", "player_receptions": "receptions",
    "player_pass_tds": "passing_tds", "player_pass_interceptions": "passing_interceptions",
}
_TD_POINTS = 6.0  # rush/rec TD value used for anytime-TD expectation


def american_to_prob(odds: float) -> float:
    return 100.0 / (odds + 100.0) if odds > 0 else (-odds) / (-odds + 100.0)


def devig_two_way(over: float, under: float) -> float:
    """Vig-removed P(over) from a two-way American market."""
    po, pu = american_to_prob(over), american_to_prob(under)
    return po / (po + pu) if (po + pu) > 0 else po


def statline_to_points(statline: dict[str, float], anytime_td_prob: float | None,
                       engine: ScoringEngine, position: str | None = None) -> float:
    """Score a prop-derived stat line; add expected rush/rec TD points from the
    de-vigged anytime-TD probability (Poisson: E[TD] = -ln(1-p))."""
    import math
    pts = engine.score_statline(statline, position)
    if anytime_td_prob and 0 < anytime_td_prob < 1:
        exp_td = -math.log(1.0 - min(anytime_td_prob, 0.95))
        pts += exp_td * _TD_POINTS
    return round(pts, 2)


class PlayerPropSource:
    name = "props"

    @staticmethod
    def enabled() -> bool:
        return bool(settings.odds_api_key)

    def weekly_points(self, season: int, week: int, league: LeagueSettings) -> dict[str, float]:
        if not self.enabled():
            return {}
        import requests

        engine = ScoringEngine(league)
        xw = crosswalk()
        markets = ",".join(list(_LINE_MARKETS) + ["player_anytime_td"])
        try:
            events = requests.get(f"{API}/events", params={"apiKey": settings.odds_api_key},
                                  timeout=20).json()
            per_player: dict[str, dict] = {}
            for ev in events:
                odds = requests.get(
                    f"{API}/events/{ev['id']}/odds",
                    params={"apiKey": settings.odds_api_key, "regions": "us",
                            "markets": markets, "oddsFormat": "american"}, timeout=20).json()
                self._collect(odds, per_player)
        except Exception as e:  # noqa: BLE001
            log.warning("Player props fetch failed (%s); source skipped.", e)
            return {}

        out: dict[str, float] = {}
        for name, data in per_player.items():
            gid = xw.resolve(name)
            if gid:
                out[gid] = statline_to_points(data["stats"], data.get("td_prob"), engine)
        return out

    def _collect(self, event_odds: dict, per_player: dict) -> None:
        for book in event_odds.get("bookmakers", []):
            for mkt in book.get("markets", []):
                canon = _LINE_MARKETS.get(mkt["key"])
                for oc in mkt.get("outcomes", []):
                    player = oc.get("description") or oc.get("name")
                    if not player:
                        continue
                    p = per_player.setdefault(player, {"stats": {}, "td_prob": None})
                    if canon and oc.get("point") is not None:
                        p["stats"].setdefault(canon, float(oc["point"]))
                    elif mkt["key"] == "player_anytime_td" and oc.get("name") == "Yes":
                        p["td_prob"] = american_to_prob(float(oc.get("price", 0)))
