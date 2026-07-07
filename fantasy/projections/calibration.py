"""In-season per-source bias calibration for the projection consensus.

Why this exists (measured, 2025 test season, 12-team half-PPR, startable rows):
Sleeper over-projected by +1.76 pts/game, so blending it RAW made the consensus
worse than our model alone (raw-source MAE 5.21 vs model 4.94). Subtracting each
source's own per-position mean error — estimated only from weeks already played —
flipped that completely: the calibrated 50/50 blend reached 4.81 MAE, beating the
model alone at every position. Consensus only works when sources are debiased.

The cardinal rule matches features.py: the bias estimate for week t may only use
weeks strictly before t. Estimates are shrunk toward zero so a hot week-1 error
doesn't overcorrect (bias * n / (n + SHRINK_K)).

Sources whose past weekly numbers can be re-fetched (Sleeper — cached parquet per
week) or recomputed (our model — features are point-in-time) are backfilled on
demand; sources that only exist live (ESPN, props) accumulate history in a small
per-season parquet store as the app projects each week.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from fantasy.config import settings

log = logging.getLogger(__name__)

SHRINK_K = 60.0  # error-rows of history at which we apply half the estimated bias


def shrunk_bias(errors: np.ndarray) -> float:
    n = len(errors)
    if n == 0:
        return 0.0
    return float(np.mean(errors)) * n / (n + SHRINK_K)


class SourceCalibrator:
    """Records per-source weekly projections and estimates per-position bias.

    Store: ``data/cache/srcproj_{season}_{scope}.parquet`` with columns
    [source, week, player_id, pts]. ``record()`` is idempotent per
    (source, week, player_id) — the first recorded number wins, so re-running a
    cycle never rewrites history with post-hoc values.

    ``scope`` must isolate leagues whose recorded values aren't comparable —
    projections are in league-scored points, so pass the scoring format
    (ppr/half_ppr/standard). Without it, one league's numbers would poison
    every other league's bias estimates in a shared multi-tenant cache.
    """

    def __init__(self, season: int, scope: str = ""):
        self.season = season
        suffix = f"_{scope}" if scope else ""
        self.path = settings.cache_dir / f"srcproj_{season}{suffix}.parquet"
        try:
            self._store = pd.read_parquet(self.path)
        except Exception as e:  # missing OR corrupt (ArrowInvalid etc.) — start fresh
            if self.path.exists():
                log.warning("Calibration store %s unreadable (%s); starting fresh.", self.path, e)
            self._store = pd.DataFrame(columns=["source", "week", "player_id", "pts"])

    def record(self, source: str, week: int, proj: dict[str, float]) -> None:
        """Persist a source's projections for a week (first write wins)."""
        if not proj:
            return
        new = pd.DataFrame(
            {"source": source, "week": int(week), "player_id": list(proj), "pts": list(proj.values())}
        )
        merged = pd.concat([self._store, new], ignore_index=True)
        merged = merged.drop_duplicates(subset=["source", "week", "player_id"], keep="first")
        if len(merged) != len(self._store):
            self._store = merged
            try:
                self._store.to_parquet(self.path, index=False)
            except Exception as e:  # read-only fs etc. — calibration degrades gracefully
                log.warning("Could not persist source projections (%s)", e)

    def past_projections(self, source: str, before_week: int) -> pd.DataFrame:
        s = self._store
        return s[(s["source"] == source) & (s["week"] < before_week)]

    def biases(self, source: str, week: int, actuals: pd.DataFrame) -> dict[str, float]:
        """Per-position shrunk bias for a source, from weeks strictly before ``week``.

        ``actuals``: current-season rows [player_id, week, position, y] for played
        weeks (the league-scored outcome).
        """
        hist = self.past_projections(source, week)
        if hist.empty or actuals.empty:
            return {}
        m = hist.merge(actuals[["player_id", "week", "position", "y"]],
                       on=["player_id", "week"], how="inner")
        if m.empty:
            return {}
        m["err"] = m["pts"] - m["y"]
        return {
            pos: shrunk_bias(sub["err"].to_numpy(dtype=float))
            for pos, sub in m.groupby("position")
        }

    def calibrate(
        self,
        sources: dict[str, dict[str, float]],
        positions: dict[str, str],
        actuals: pd.DataFrame,
        week: int,
    ) -> dict[str, dict[str, float]]:
        """Return sources with each one's per-position bias subtracted (floored at 0)."""
        out: dict[str, dict[str, float]] = {}
        for name, proj in sources.items():
            bias = self.biases(name, week, actuals)
            if bias:
                shifts = {p: round(b, 2) for p, b in bias.items() if abs(b) >= 0.05}
                if shifts:
                    log.info("Calibrating %s (week %d): %s", name, week, shifts)
            out[name] = {
                pid: max(pts - bias.get(positions.get(pid, ""), 0.0), 0.0)
                for pid, pts in proj.items()
            }
        return out
