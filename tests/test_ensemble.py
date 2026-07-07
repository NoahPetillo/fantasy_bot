"""StackedBlender: non-negative weights, finite predictions, no NaN/inf leakage."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fantasy.projections.ensemble import StackedBlender


def _synth(n=400, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.choice(["RB", "WR"], size=n)
    truth = rng.gamma(3, 4, size=n)
    df = pd.DataFrame(
        {
            "position": pos,
            "y": truth,
            "proj": truth + rng.normal(0, 3, n),
            "pts_trail_mean": truth + rng.normal(0, 5, n),
            "pts_season_mean": truth + rng.normal(0, 4, n),
            "pts_last": truth + rng.normal(0, 7, n),
            "pts_ewm": truth + rng.normal(0, 5, n),
        }
    )
    return df


def test_weights_nonnegative_and_predictions_finite():
    df = _synth()
    b = StackedBlender().fit(df)
    assert b.global_weights is not None
    assert (b.global_weights >= -1e-9).all()
    pred = b.predict(df)
    assert np.isfinite(pred.to_numpy()).all()
    assert (pred >= 0).all()


def test_blend_not_worse_than_best_component_in_sample():
    df = _synth()
    b = StackedBlender().fit(df)
    blend_mae = (b.predict(df) - df["y"]).abs().mean()
    comp_maes = [(df[c] - df["y"]).abs().mean() for c in b.components]
    # In-sample, the NNLS blend should be <= the best single component.
    assert blend_mae <= min(comp_maes) + 1e-6


def test_handles_nan_components():
    df = _synth()
    df.loc[df.index[:20], "pts_last"] = np.nan  # week-1-style gaps
    b = StackedBlender().fit(df)
    assert np.isfinite(b.predict(df).to_numpy()).all()
