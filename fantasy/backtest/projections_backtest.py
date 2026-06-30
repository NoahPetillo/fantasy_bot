"""Walk-forward backtest for the projection model.

Trains on prior seasons, predicts a held-out season, and reports leak-free
accuracy (MAE/RMSE/bias/corr) by position against strong naive baselines:

- last        : last game's points (lag-1)
- trail4      : trailing 4-game mean
- season_avg  : season-to-date mean

A projection model only earns trust if it beats trailing-average — many do not.
(The head-to-head vs ESPN's own projections happens live in Phase 2 using ESPN's
``projected_points`` from the read API; offline we hold the bar at trailing-avg.)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from fantasy.data.nfl import load_weekly
from fantasy.projections.ensemble import StackedBlender
from fantasy.projections.features import build_features
from fantasy.projections.model import POSITIONS, ProjectionModel

log = logging.getLogger(__name__)

BASELINES = {"last": "pts_last", "trail4": "pts_trail_mean", "season_avg": "pts_season_mean"}


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    valid = ~np.isnan(y_pred)
    err = err[valid]
    if len(err) == 0:
        return dict(n=0, mae=np.nan, rmse=np.nan, bias=np.nan, corr=np.nan)
    return dict(
        n=int(len(err)),
        mae=float(np.mean(np.abs(err))),
        rmse=float(np.sqrt(np.mean(err**2))),
        bias=float(np.mean(err)),
        corr=float(np.corrcoef(y_true[valid], y_pred[valid])[0, 1]) if len(err) > 1 else np.nan,
    )


def run_backtest(
    train_seasons: list[int],
    test_season: int,
    scoring_engine,
    relevance_threshold: float = 3.0,
    max_week: int | None = None,
) -> tuple[pd.DataFrame, ProjectionModel, StackedBlender]:
    """Train on ``train_seasons``, evaluate on ``test_season``.

    Only rows where the player has prior history and trailing form
    >= ``relevance_threshold`` count toward metrics (so the score reflects
    startable players, not deep-bench zeros). Features are point-in-time, so
    training on whole prior seasons + predicting the test season is leak-free.

    Also fits a StackedBlender on a held-out validation season (the last training
    season) and reports the blended prediction alongside the raw model.
    """
    all_seasons = sorted(set(train_seasons) | {test_season})
    weekly = load_weekly(all_seasons)
    feat = build_features(weekly, scoring_engine)

    train = feat[feat["season"].isin(train_seasons)].copy()
    test = feat[feat["season"] == test_season].copy()
    if max_week is not None:
        test = test[test["week"] <= max_week]

    # Fit the stacking blender on a held-out validation season (leak-free):
    # temp model trained on earlier seasons -> predict the last train season -> NNLS.
    blender = StackedBlender()
    train_sorted = sorted(train_seasons)
    if len(train_sorted) >= 2:
        fit_seasons, val_season = train_sorted[:-1], train_sorted[-1]
        val = feat[feat["season"] == val_season].copy()
        tmp = ProjectionModel().fit(feat[feat["season"].isin(fit_seasons)])
        val["proj"] = tmp.predict(val)
        val = val[(val["games_so_far"] >= 1) & (val["pts_trail_mean"] >= relevance_threshold)]
        blender.fit(val)

    model = ProjectionModel().fit(train)
    test["proj"] = model.predict(test)
    test["blend"] = blender.predict_fast(test) if blender.global_weights is not None else test["proj"]

    # Evaluate on startable rows only.
    evalset = test[(test["games_so_far"] >= 1) & (test["pts_trail_mean"] >= relevance_threshold)]

    rows = []
    for pos in POSITIONS + ["ALL"]:
        sub = evalset if pos == "ALL" else evalset[evalset["position"] == pos]
        if sub.empty:
            continue
        y = sub["y"].to_numpy(dtype=float)
        rec = {"position": pos, "n": len(sub)}
        m = _metrics(y, sub["proj"].to_numpy(dtype=float))
        b = _metrics(y, sub["blend"].to_numpy(dtype=float))
        rec.update({"model_mae": m["mae"], "blend_mae": b["mae"],
                    "blend_bias": b["bias"], "blend_corr": b["corr"]})
        for name, col in BASELINES.items():
            bm = _metrics(y, sub[col].to_numpy(dtype=float))
            rec[f"{name}_mae"] = bm["mae"]
        rec["model_vs_trail4_%"] = round(100 * (rec["trail4_mae"] - rec["model_mae"]) / rec["trail4_mae"], 1)
        rec["blend_vs_trail4_%"] = round(100 * (rec["trail4_mae"] - rec["blend_mae"]) / rec["trail4_mae"], 1)
        rows.append(rec)

    report = pd.DataFrame(rows).set_index("position")
    return report, model, blender


def format_report(report: pd.DataFrame, test_season: int) -> str:
    cols = ["n", "trail4_mae", "season_avg_mae", "model_mae", "blend_mae",
            "model_vs_trail4_%", "blend_vs_trail4_%", "blend_corr"]
    cols = [c for c in cols if c in report.columns]
    body = report[cols].round(2).to_string()
    won = report.loc[report.index != "ALL", "blend_vs_trail4_%"]
    verdict = (
        f"Blend beats trailing-4 on {int((won > 0).sum())}/{len(won)} positions; "
        f"overall blend MAE improvement vs trail4: {report.loc['ALL', 'blend_vs_trail4_%']:.1f}% "
        f"(raw model: {report.loc['ALL', 'model_vs_trail4_%']:.1f}%)"
        if "ALL" in report.index else ""
    )
    return f"=== Projection backtest — test season {test_season} (startable rows) ===\n{body}\n{verdict}"
