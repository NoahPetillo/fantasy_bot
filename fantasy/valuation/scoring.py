"""ScoringEngine — turns a raw stat line into fantasy points using the LEAGUE'S
own scoring rules.

One engine is built from :class:`fantasy.league_settings.LeagueSettings` and used
everywhere points are needed:

- backtests: score historical nflverse stat lines (so past seasons match the league)
- projections: convert projected stat lines to projected points
- what-if: re-score a player under the league's exact rules

Because it is driven entirely by ``LeagueSettings.scoring`` (a canonical
stat -> points map) plus optional TE-premium and DST tiers, it adapts to PPR /
half / standard / 6-pt-pass-TD / PPFD / TE-premium / IDP / custom with no code
changes.
"""

from __future__ import annotations

import logging

import pandas as pd

from fantasy.espn.stat_ids import CANONICAL_TO_NFLVERSE
from fantasy.league_settings import LeagueSettings

log = logging.getLogger(__name__)


class ScoringEngine:
    def __init__(self, settings: LeagueSettings):
        self.settings = settings
        self.scoring = dict(settings.scoring)
        self.reception_bonus = dict(settings.position_reception_bonus)
        self._warned_missing: set[str] = set()

    # ── single stat line ──────────────────────────────────────────────────────
    def score_statline(self, stats: dict[str, float], position: str | None = None) -> float:
        """Score one canonical stat line (keys are canonical stat names)."""
        total = 0.0
        for stat, value in stats.items():
            pts = self.scoring.get(stat)
            if pts:
                total += value * pts
        # TE-premium (or any position-specific reception bonus).
        if position and position in self.reception_bonus:
            total += stats.get("receptions", 0.0) * self.reception_bonus[position]
        # DST tiered points-allowed, if the league scores it and we have the value.
        if position == "D/ST":
            total += self._dst_points_allowed_score(stats.get("points_allowed"))
        return round(total, 2)

    def _dst_points_allowed_score(self, points_allowed: float | None) -> float:
        if points_allowed is None:
            return 0.0
        pa = points_allowed
        if pa == 0:
            key = "dst_points_allowed_0"
        elif pa <= 6:
            key = "dst_points_allowed_1_6"
        elif pa <= 13:
            key = "dst_points_allowed_7_13"
        elif pa <= 17:
            key = "dst_points_allowed_14_17"
        elif pa <= 27:
            key = "dst_points_allowed_18_27"
        elif pa <= 34:
            key = "dst_points_allowed_28_34"
        elif pa <= 45:
            key = "dst_points_allowed_35_45"
        else:
            key = "dst_points_allowed_46plus"
        return self.scoring.get(key, 0.0)

    # ── vectorized over a nflverse weekly DataFrame ───────────────────────────
    def score_dataframe(self, df: pd.DataFrame, position_col: str = "position") -> pd.Series:
        """Compute fantasy points for an entire nflverse weekly stats frame.

        Maps each canonical scoring stat to its nflverse column and accumulates
        ``column * points``. Unknown/missing columns contribute 0 and are warned
        once. Adds TE-premium per ``position_col`` if configured.
        """
        points = pd.Series(0.0, index=df.index)
        for canonical, pts in self.scoring.items():
            if not pts:
                continue
            col = CANONICAL_TO_NFLVERSE.get(canonical)
            if col is None:
                self._warn_missing(canonical, reason="no nflverse mapping")
                continue
            if col not in df.columns:
                self._warn_missing(col, reason="column absent from frame")
                continue
            points = points.add(df[col].fillna(0.0) * pts, fill_value=0.0)

        if self.reception_bonus and "receptions" in df.columns and position_col in df.columns:
            for pos, bonus in self.reception_bonus.items():
                mask = df[position_col] == pos
                points.loc[mask] = points.loc[mask] + df.loc[mask, "receptions"].fillna(0.0) * bonus

        return points.round(2)

    def _warn_missing(self, name: str, reason: str) -> None:
        if name not in self._warned_missing:
            self._warned_missing.add(name)
            log.warning("ScoringEngine: stat '%s' not scored (%s)", name, reason)

    # ── convenience ───────────────────────────────────────────────────────────
    def __repr__(self) -> str:
        nz = {k: v for k, v in self.scoring.items() if v}
        return f"ScoringEngine({self.settings.scoring_format.value}, {len(nz)} active rules)"
