"""Project a DISTRIBUTION, not just a mean.

Weekly fantasy scoring is right-skewed and non-negative, so we model each
player-week as a Gamma whose mean is the model projection and whose spread comes
from the residual volatility at that projection level / position. This is what
makes win-probability start/sit and risk-aware decisions possible later (sampling
lineups, computing P(win), ceiling/floor).

We estimate the spread empirically: bin backtest residuals by position and
projection level, store the residual std, and map (position, mean) -> sd.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats


def gamma_from_mean_sd(mean: float, sd: float) -> stats._distn_infrastructure.rv_frozen:
    """Frozen Gamma with the given mean and sd (shape=k, scale=theta)."""
    mean = max(mean, 1e-6)
    sd = max(sd, 1e-3)
    k = (mean / sd) ** 2
    theta = sd**2 / mean
    return stats.gamma(a=k, scale=theta)


@dataclass
class VarianceModel:
    """Maps (position, projected_mean) -> residual standard deviation.

    Fit on out-of-sample residuals (proj - actual). Within a position we let sd
    grow with the projection via a simple linear fit sd = a + b*mean, floored.
    """

    coeffs: dict[str, tuple[float, float]] = field(default_factory=dict)
    floor: dict[str, float] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame, proj_col: str = "proj", actual_col: str = "y") -> "VarianceModel":
        for pos, sub in df.dropna(subset=[proj_col]).groupby("position"):
            resid = (sub[actual_col] - sub[proj_col]).to_numpy(dtype=float)
            mean = sub[proj_col].to_numpy(dtype=float)
            # abs residual ~ a + b*mean  (proxy for sd scaling with usage)
            if len(sub) >= 30 and np.ptp(mean) > 1e-6:
                b, a = np.polyfit(mean, np.abs(resid), 1)
            else:
                a, b = float(np.std(resid)), 0.0
            self.coeffs[pos] = (float(a), float(b))
            self.floor[pos] = float(max(np.std(resid) * 0.5, 1.0))
        return self

    def sd(self, position: str, mean: float) -> float:
        a, b = self.coeffs.get(position, (max(0.5 * mean, 3.0), 0.0))
        return max(a + b * max(mean, 0.0), self.floor.get(position, 1.0))

    def gamma(self, position: str, mean: float):
        return gamma_from_mean_sd(mean, self.sd(position, mean))

    def quantiles(self, position: str, mean: float, qs=(0.1, 0.5, 0.9)) -> dict[float, float]:
        g = self.gamma(position, mean)
        return {q: float(g.ppf(q)) for q in qs}
