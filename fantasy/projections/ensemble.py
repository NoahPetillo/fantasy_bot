"""Stacked blend of the model with naive baselines.

Per the research, a source-level blend reliably beats any single projection
source. We learn non-negative weights (NNLS) over component predictions on a
held-out validation season, so the blend can lean on the model where it helps and
fall back to trailing/season averages where it doesn't (e.g. noisy TEs). Weights
are fit per position, since the right mix differs by position.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import nnls

# Component prediction columns to blend (model proj + point-in-time baselines).
COMPONENTS = ["proj", "pts_trail_mean", "pts_season_mean", "pts_last"]


class StackedBlender:
    def __init__(self, components: list[str] | None = None):
        self.components = components or COMPONENTS
        self.weights: dict[str, np.ndarray] = {}
        self.global_weights: np.ndarray | None = None

    def _design(self, df: pd.DataFrame) -> np.ndarray:
        return df[self.components].fillna(0.0).to_numpy(dtype=float)

    def fit(self, val: pd.DataFrame, target: str = "y") -> "StackedBlender":
        y_all = val[target].to_numpy(dtype=float)
        self.global_weights, _ = nnls(self._design(val), y_all)
        for pos, sub in val.groupby("position"):
            if len(sub) >= 50:
                w, _ = nnls(self._design(sub), sub[target].to_numpy(dtype=float))
                self.weights[pos] = w
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Apply per-position blend weights (vectorized elementwise, inf/nan-safe)."""
        if self.global_weights is None:
            return df.get("proj", pd.Series(0.0, index=df.index)).clip(lower=0.0)
        X = np.nan_to_num(self._design(df), nan=0.0, posinf=0.0, neginf=0.0)
        # Per-row weight matrix: position weights where known, else global.
        W = np.tile(self.global_weights, (len(df), 1))
        pos = df["position"].to_numpy()
        for p, w in self.weights.items():
            W[pos == p] = w
        out = (X * W).sum(axis=1)
        return pd.Series(out, index=df.index).clip(lower=0.0)

    # Backwards-compatible alias.
    predict_fast = predict

    def describe(self) -> str:
        lines = []
        for pos, w in self.weights.items():
            terms = ", ".join(f"{c}:{wi:.2f}" for c, wi in zip(self.components, w))
            lines.append(f"  {pos}: {terms}")
        return "Blend weights (per position):\n" + "\n".join(lines)
