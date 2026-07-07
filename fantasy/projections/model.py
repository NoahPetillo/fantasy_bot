"""Per-position gradient-boosted projection models.

A separate XGBoost regressor per position (QB/RB/WR/TE) — usage and scoring
dynamics differ enough that one model per position beats a shared one. Targets
the LEAGUE-scored weekly points (so projections are already in the league's
currency). XGBoost ingests NaN trailing features natively (rookies / week 1).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from fantasy.projections.features import feature_columns

log = logging.getLogger(__name__)

POSITIONS = ["QB", "RB", "WR", "TE"]

_DEFAULT_PARAMS = dict(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5.0,
    reg_lambda=1.0,
    objective="reg:squarederror",
    n_jobs=-1,
    random_state=13,
)

# Per-position hyperparameters from a randomized search (train 2021-2023, early
# stopping on 2024 startable rows; verified out-of-sample on 2025 and 2023).
# Positions differ: WR wants shallow/heavily-regularized trees, RB deeper ones.
_POSITION_PARAMS: dict[str, dict] = {
    "QB": dict(n_estimators=119, max_depth=4, learning_rate=0.03,
               min_child_weight=1.0, subsample=0.7, colsample_bytree=1.0, reg_lambda=1.0),
    "RB": dict(n_estimators=185, max_depth=5, learning_rate=0.02,
               min_child_weight=10.0, subsample=0.9, colsample_bytree=1.0, reg_lambda=10.0),
    "WR": dict(n_estimators=56, max_depth=3, learning_rate=0.05,
               min_child_weight=20.0, subsample=0.7, colsample_bytree=0.8, reg_lambda=1.0),
    "TE": dict(n_estimators=125, max_depth=3, learning_rate=0.02,
               min_child_weight=5.0, subsample=0.7, colsample_bytree=0.6, reg_lambda=10.0),
}


def position_params(position: str) -> dict:
    """Tuned hyperparameters for a position, falling back to the defaults."""
    return {**_DEFAULT_PARAMS, **_POSITION_PARAMS.get(position, {})}

# Training rows from season s get weight SEASON_DECAY ** (latest_season - s):
# the game drifts (rules, scheme trends, player pool), so recent seasons matter more.
SEASON_DECAY = 0.85


@dataclass
class PositionModel:
    position: str
    params: dict = field(default_factory=lambda: dict(_DEFAULT_PARAMS))
    season_decay: float = SEASON_DECAY
    model: XGBRegressor | None = None
    features: list[str] = field(default_factory=list)

    def fit(self, feat: pd.DataFrame) -> "PositionModel":
        sub = feat[feat["position"] == self.position]
        self.features = feature_columns(feat)
        X = sub[self.features].astype(float)
        y = sub["y"].astype(float)
        w = None
        if self.season_decay and self.season_decay < 1.0:
            age = sub["season"].max() - sub["season"]
            w = np.power(self.season_decay, age.astype(float))
        self.model = XGBRegressor(**self.params)
        self.model.fit(X, y, sample_weight=w)
        log.info("Trained %s model on %d rows, %d features", self.position, len(sub), len(self.features))
        return self

    def predict(self, feat: pd.DataFrame) -> np.ndarray:
        assert self.model is not None, "model not fit"
        X = feat[self.features].astype(float)
        # Projected points can't be negative for skill positions; clip at 0.
        return np.clip(self.model.predict(X), 0.0, None)

    def importances(self, top: int = 15) -> pd.Series:
        assert self.model is not None
        return (
            pd.Series(self.model.feature_importances_, index=self.features)
            .sort_values(ascending=False)
            .head(top)
        )


class ProjectionModel:
    """Holds one PositionModel per position and projects a whole feature table."""

    def __init__(self, params: dict | None = None):
        # Explicit params override the tuned per-position sets for every position.
        self.params = params
        self.models: dict[str, PositionModel] = {}

    def fit(self, feat: pd.DataFrame) -> "ProjectionModel":
        for pos in POSITIONS:
            if (feat["position"] == pos).any():
                p = dict(self.params) if self.params else position_params(pos)
                self.models[pos] = PositionModel(pos, p).fit(feat)
        return self

    def predict(self, feat: pd.DataFrame) -> pd.Series:
        out = pd.Series(np.nan, index=feat.index, name="proj")
        for pos, m in self.models.items():
            mask = feat["position"] == pos
            if mask.any():
                out.loc[mask] = m.predict(feat.loc[mask])
        return out
