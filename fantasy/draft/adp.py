"""ADP (average draft position) + survival probability — the opponent model.

Opponents are modeled as drafting near each player's ADP with noise. From that we
compute the survival probability that a player is still available at our next pick
— the core signal for the now-or-never draft logic. Source: Fantasy Football
Calculator's free JSON API (true ADP + stdev + high/low; historical via ?year=).
"""

from __future__ import annotations

import logging

import pandas as pd
import requests
from scipy.stats import norm

from fantasy.config import settings
from fantasy.data.ids import crosswalk

log = logging.getLogger(__name__)

FFC = "https://fantasyfootballcalculator.com/api/v1/adp/{fmt}"
UA = {"User-Agent": "Mozilla/5.0"}


def load_ffc_adp(season: int | None, teams: int = 12, fmt: str = "ppr",
                 refresh: bool = False) -> pd.DataFrame:
    """ADP table mapped to gsis ids. ``season=None`` -> current draft season.

    Columns: player_id (gsis or synthetic DST:/K:), name, position, team, adp,
    sd, high, low, times_drafted.
    """
    key = f"adp_{fmt}_{teams}_{season or 'current'}"
    path = settings.cache_dir / f"{key}.parquet"
    if not refresh and path.exists():
        return pd.read_parquet(path)

    params = {"teams": teams}
    if season is not None:
        params["year"] = season
    r = requests.get(FFC.format(fmt=fmt), params=params, headers=UA, timeout=20)
    r.raise_for_status()
    players = r.json().get("players", [])
    xw = crosswalk()

    rows = []
    for p in players:
        pos = p.get("position")
        gid = _map_id(p, pos, xw)
        rows.append({
            "player_id": gid, "name": p.get("name"), "position": pos, "team": p.get("team"),
            "adp": float(p.get("adp", 0) or 0), "sd": float(p.get("stdev", 0) or 0),
            "high": p.get("high"), "low": p.get("low"),
            "times_drafted": int(p.get("times_drafted", 0) or 0),
        })
    df = pd.DataFrame(rows)
    unmatched = df["player_id"].isna().sum()
    if unmatched:
        log.warning("ADP: %d/%d players unmatched to gsis (likely rookies/D-ST/K).",
                    unmatched, len(df))
    df.to_parquet(path, index=False)
    return df


def _map_id(p: dict, pos: str, xw) -> str | None:
    if pos == "DST":
        return f"DST:{p.get('team')}"
    if pos == "PK" or pos == "K":
        gid = xw.resolve(p.get("name", ""), "K")
        return gid or f"K:{p.get('name')}"
    return xw.resolve(p.get("name", ""), pos)


def survival(adp: float, sd: float, next_pick: int, current_pick: int | None = None) -> float:
    """P(player lasts to ``next_pick``), optionally conditioned on still being
    available at ``current_pick`` (so elite players that 'fall' update correctly).

    A latent draft slot ~ Normal(adp, sd) with an early-season sd floor to avoid
    overconfident 0/1 probabilities.
    """
    sigma = max(sd, 0.5 * (adp ** 0.5), 1.0)
    p_after_next = 1.0 - norm.cdf(next_pick, loc=adp, scale=sigma)
    if current_pick is None or current_pick <= 0:
        return float(min(max(p_after_next, 0.0), 1.0))
    p_after_cur = 1.0 - norm.cdf(current_pick, loc=adp, scale=sigma)
    if p_after_cur <= 1e-9:
        return 1.0  # already past where they'd normally go -> very likely to keep falling
    return float(min(max(p_after_next / p_after_cur, 0.0), 1.0))
