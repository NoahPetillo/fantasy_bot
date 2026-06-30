"""The real 'beat ESPN' gate — our weekly projections vs ESPN's own.

Uses YOUR league's box scores for 2025, which carry, per player per week, both
ESPN's projected_points AND the actual points scored under your exact league
scoring (the ground truth). We train our model on 2021-2024, project every 2025
week, join on the same players, and compare MAE/RMSE by position.

    uv run python scripts/headtohead_espn.py
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from fantasy.data.ids import crosswalk
from fantasy.data.nfl import load_weekly
from fantasy.espn.client import EspnClient
from fantasy.projections.features import build_features
from fantasy.projections.service import ProjectionService

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
pd.set_option("display.width", 140)

TRAIN = [2021, 2022, 2023, 2024]
TEST = 2025
WEEKS = range(1, 18)
SKILL = {"QB", "RB", "WR", "TE"}
START_THRESHOLD = 6.0  # ESPN-projected pts; focus on startable players


def collect_espn(client) -> pd.DataFrame:
    """ESPN projected + actual per skill player per week, keyed by gsis id."""
    xw = crosswalk()
    rows = []
    for wk in WEEKS:
        try:
            box = client.box_scores(wk)
        except Exception as e:  # noqa: BLE001
            logging.warning("box_scores wk%s failed: %s", wk, e)
            continue
        for m in box:
            for bp in (getattr(m, "home_lineup", []) or []) + (getattr(m, "away_lineup", []) or []):
                pos = getattr(bp, "position", None)
                if pos not in SKILL:
                    continue
                gid = xw.from_espn(getattr(bp, "playerId", None))
                rows.append({
                    "week": wk, "player_id": gid, "position": pos,
                    "espn_proj": getattr(bp, "projected_points", None),
                    "actual": getattr(bp, "points", None),
                })
    return pd.DataFrame(rows)


def our_projections(service: ProjectionService) -> pd.DataFrame:
    feat = build_features(load_weekly([*TRAIN, TEST]), service.engine)
    cur = feat[feat["season"] == TEST].copy()
    cur["proj"] = service.model.predict(cur)
    if service.blender.global_weights is not None:
        cur["proj"] = service.blender.predict(cur)
    return cur[["player_id", "week", "proj", "pts_trail_mean"]].rename(columns={"proj": "our_proj"})


def sleeper_projections(league) -> pd.DataFrame:
    from fantasy.projections.sources import SleeperProjectionSource
    src = SleeperProjectionSource()
    rows = []
    for wk in WEEKS:
        for pid, pts in src.weekly_points(TEST, wk, league).items():
            rows.append({"player_id": pid, "week": wk, "sleeper": pts})
    return pd.DataFrame(rows)


def consensus_vs_espn(df: pd.DataFrame) -> None:
    import numpy as np
    d = df.copy()
    # equal-weight mean over available sources per row
    d["consensus"] = d[["our_proj", "espn_proj", "sleeper"]].mean(axis=1, skipna=True)
    d = d.dropna(subset=["consensus", "actual"])
    rows = []
    for pos in ["QB", "RB", "WR", "TE", "ALL"]:
        sub = d if pos == "ALL" else d[d["position"] == pos]
        if sub.empty:
            continue
        em, cm = mae(sub["espn_proj"], sub["actual"]), mae(sub["consensus"], sub["actual"])
        rows.append({"position": pos, "n": len(sub), "espn_mae": round(em, 2),
                     "consensus_mae": round(cm, 2),
                     "edge_%": round(100 * (em - cm) / em, 1),
                     "winner": "CONSENSUS" if cm < em else "ESPN"})
    rep = pd.DataFrame(rows).set_index("position")
    print("\n=== Multi-source consensus (model+ESPN+Sleeper) vs ESPN alone (2025) ===")
    print(rep.to_string())


def blend_vs_espn(startable: pd.DataFrame) -> None:
    """Ensemble our model WITH ESPN's projection; test out-of-sample.

    Fit non-negative blend weights on early weeks (1-9), evaluate on later weeks
    (10+). This is what the live system does — ESPN's projection is a free input.
    """
    from scipy.optimize import nnls

    cols = ["our_proj", "espn_proj", "pts_trail_mean"]
    tr = startable[startable["week"] <= 9].dropna(subset=cols + ["actual"])
    te = startable[startable["week"] >= 10].dropna(subset=cols + ["actual"])
    if len(tr) < 50 or te.empty:
        return
    w, _ = nnls(tr[cols].to_numpy(float), tr["actual"].to_numpy(float))
    blend = np.clip(sum(w[i] * te[c].to_numpy(float) for i, c in enumerate(cols)), 0, None)
    bm, em = mae(pd.Series(blend, index=te.index), te["actual"]), mae(te["espn_proj"], te["actual"])
    edge = 100 * (em - bm) / em
    print("\n=== Ensemble (our model + ESPN proj), out-of-sample weeks 10-17 ===")
    print(f"  blend weights  our:{w[0]:.2f}  espn:{w[1]:.2f}  trailing:{w[2]:.2f}")
    print(f"  ESPN MAE {em:.2f}  |  our+ESPN blend MAE {bm:.2f}  |  edge {edge:+.1f}%")
    print(f"  → {'Blend BEATS ESPN' if bm < em else 'ESPN still ahead'} on held-out weeks.")


def mae(a, b):
    d = (a - b).abs()
    return float(d.mean())


def main() -> int:
    client = EspnClient(season=TEST)
    league = client.league_settings()
    print(f"League: {league.summary()}\nTraining our model on {TRAIN}, testing {TEST}...\n")

    service = ProjectionService(league).fit(TRAIN)
    espn = collect_espn(client)
    ours = our_projections(service)

    df = espn.merge(ours, on=["player_id", "week"], how="inner").dropna(
        subset=["espn_proj", "actual", "our_proj"]
    )
    df = df[df["player_id"].notna()]
    df = df.merge(sleeper_projections(league), on=["player_id", "week"], how="left")
    startable = df[df["espn_proj"] >= START_THRESHOLD]
    print(f"Joined player-weeks: {len(df):,} | startable (ESPN proj ≥ {START_THRESHOLD}): {len(startable):,}\n")

    rows = []
    for pos in ["QB", "RB", "WR", "TE", "ALL"]:
        sub = startable if pos == "ALL" else startable[startable["position"] == pos]
        if sub.empty:
            continue
        em, om = mae(sub["espn_proj"], sub["actual"]), mae(sub["our_proj"], sub["actual"])
        rows.append({
            "position": pos, "n": len(sub),
            "espn_mae": round(em, 2), "our_mae": round(om, 2),
            "our_edge_%": round(100 * (em - om) / em, 1),
            "winner": "OURS" if om < em else "ESPN",
        })
    report = pd.DataFrame(rows).set_index("position")
    print("=== Head-to-head: our model vs ESPN projections (2025, your scoring) ===")
    print(report.to_string())
    overall = report.loc["ALL"] if "ALL" in report.index else None
    if overall is not None:
        verb = "BEATS" if overall["winner"] == "OURS" else "trails"
        print(f"\nStandalone: our model {verb} ESPN by {overall['our_edge_%']:+.1f}% MAE "
              f"(ours {overall['our_mae']} vs ESPN {overall['espn_mae']}).")

    consensus_vs_espn(startable)
    blend_vs_espn(startable)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
