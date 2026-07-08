"""Season draft-value board — the per-stat projection engine for the draft plan.

Builds one cross-positional value board for a season, scored under THIS league's
exact rules (so 0.25/target, individual return yards, IDP and HC value all fall
out automatically). Pipeline:

1. Merge ESPN-kona + Sleeper per-stat *offense* projections on gsis (mean where
   both project a stat; ESPN is the only targets source), score each row with the
   league :class:`~fantasy.valuation.scoring.ScoringEngine`.
2. Add the returner overlay (:func:`fantasy.data.returns.return_points_overlay`).
3. Add IDP rows from Sleeper's defensive projections, scored the same way.
4. Add HC rows from :func:`fantasy.valuation.hc.hc_draft_ev`.
5. Merge FFC ADP (+ Sleeper ADP for IDP) for the opponent/survival model.
6. Compute cross-positional VOR (:func:`fantasy.valuation.vor.compute_vor`).

If neither source has published season projections yet (preseason gap), it falls
back to prior-season realized points as pseudo-projections — the board ALWAYS
builds, and improves as sources publish.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from fantasy.data.returns import OVERLAY_OWNED_STATS, return_points_overlay
from fantasy.draft.adp import load_ffc_adp
from fantasy.espn.client import EspnClient
from fantasy.espn.stat_ids import IDP_POSITIONS
from fantasy.league_settings import LeagueSettings, ScoringFormat
from fantasy.projections.sources import sleeper_season_projections
from fantasy.valuation.hc import hc_draft_ev
from fantasy.valuation.scoring import ScoringEngine
from fantasy.valuation.vor import compute_vor, pooled_position, replacement_counts

log = logging.getLogger(__name__)

# Canonical stat columns that either source may project (everything the
# ScoringEngine might read). Non-stat/id columns are excluded from the merge-mean.
_META_COLS = {"espn_id", "sleeper_id", "player_id", "name", "position", "team",
              "adp", "auction_value", "rank_ppr", "rank_std",
              "adp_half_ppr", "adp_ppr", "adp_std", "adp_idp"}
# ESPN is the authoritative (only) source of projected targets.
_ESPN_ONLY_STATS = {"receiving_targets"}
# Sleeper position labels for individual defenders we surface as IDP board rows.
_IDP_SOURCE_POSITIONS = IDP_POSITIONS | {"DB", "DL", "S", "CB", "DE", "DT", "LB"}
# Granular Sleeper/ESPN defensive labels -> our IDP tokens (vor pools them anyway).
_IDP_NORMALIZE = {"ILB": "LB", "MLB": "LB", "OLB": "LB", "EDGE": "DE",
                  "FS": "S", "SS": "S", "NT": "DT"}
_DEFAULT_ADP = 250.0
_DEFAULT_ADP_SD = 20.0
# Rough overall-pick anchor to place IDP/HC ADP when only Sleeper's positional ADP
# is available (keeps them out of the early rounds of the survival model).
_IDP_ADP_FLOOR = 150.0
_HC_ADP = 240.0


def _fmt_for_league(league: LeagueSettings) -> str:
    """FFC ADP format string from the league's reception scoring."""
    fmt = league.scoring_format
    if fmt == ScoringFormat.standard:
        return "standard"
    if fmt == ScoringFormat.half_ppr:
        return "half-ppr"
    return "ppr"


def _stat_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _META_COLS]


def _merge_offense(espn: pd.DataFrame, sleeper: pd.DataFrame) -> pd.DataFrame:
    """Outer-merge the two per-stat frames on gsis; mean where both project a stat.

    Only rows with a gsis id participate (synthetic/unmatched rows can't be joined
    cleanly). ESPN wins for targets; otherwise stats are averaged when both exist.
    """
    e = espn[espn["player_id"].notna()].copy() if not espn.empty else espn
    s = sleeper[sleeper["player_id"].notna()].copy() if not sleeper.empty else sleeper
    # Offense only on BOTH sides: IDP rows are added separately (from Sleeper,
    # the only source projecting defensive stats). If ESPN's defender rows —
    # which carry NO stat line — entered the merge, the later drop_duplicates
    # would keep the empty ESPN row and shadow the real IDP projection.
    if not e.empty:
        e = e[~e["position"].isin(_IDP_SOURCE_POSITIONS) | e["position"].isna()].copy()
    if not s.empty:
        s = s[~s["position"].isin(_IDP_SOURCE_POSITIONS) | s["position"].isna()].copy()

    if e.empty and s.empty:
        return pd.DataFrame(columns=["player_id", "name", "position", "team"])
    if s.empty:
        return e.drop_duplicates("player_id").reset_index(drop=True)
    if e.empty:
        return s.drop_duplicates("player_id").reset_index(drop=True)

    e = e.drop_duplicates("player_id").set_index("player_id")
    s = s.drop_duplicates("player_id").set_index("player_id")
    idx = e.index.union(s.index)
    out = pd.DataFrame(index=idx)
    # Identity columns: prefer ESPN, fall back to Sleeper.
    for col in ("name", "position", "team"):
        out[col] = e[col].reindex(idx) if col in e.columns else np.nan
        if col in s.columns:
            out[col] = out[col].fillna(s[col].reindex(idx))

    stat_cols = set(_stat_columns(e.reset_index())) | set(_stat_columns(s.reset_index()))
    for col in stat_cols:
        ev = e[col].reindex(idx) if col in e.columns else pd.Series(np.nan, index=idx)
        sv = s[col].reindex(idx) if col in s.columns else pd.Series(np.nan, index=idx)
        if col in _ESPN_ONLY_STATS:
            out[col] = ev
        else:
            stacked = pd.concat([ev, sv], axis=1)
            out[col] = stacked.mean(axis=1, skipna=True)
    return out.reset_index().rename(columns={"index": "player_id"})


def _score_rows(df: pd.DataFrame, engine: ScoringEngine) -> pd.Series:
    """Score every row's canonical stat columns under the league engine.

    Overlay-owned stats (return yardage) are excluded — fantasy.data.returns is
    their single source of points, so a projection source that starts publishing
    them can never double-count with the overlay."""
    if df.empty:
        return pd.Series(dtype=float)
    stat_cols = [c for c in _stat_columns(df) if c not in OVERLAY_OWNED_STATS]
    proj = pd.Series(0.0, index=df.index)
    # Sum stat*points across the columns the engine scores.
    for col in stat_cols:
        pts = engine.scoring.get(col)
        if pts:
            proj = proj.add(pd.to_numeric(df[col], errors="coerce").fillna(0.0) * pts,
                            fill_value=0.0)
    # Position reception bonus (TE-premium etc.).
    if engine.reception_bonus and "receptions" in df.columns and "position" in df.columns:
        for pos, bonus in engine.reception_bonus.items():
            mask = df["position"] == pos
            proj.loc[mask] = proj.loc[mask] + pd.to_numeric(
                df.loc[mask, "receptions"], errors="coerce").fillna(0.0) * bonus
    return proj.round(2)


def _idp_rows(sleeper: pd.DataFrame, engine: ScoringEngine) -> pd.DataFrame:
    """IDP board rows from Sleeper's defensive projections, scored by the engine."""
    if sleeper.empty:
        return pd.DataFrame(columns=["player_id", "name", "position", "team", "proj"])
    idp = sleeper[sleeper["position"].isin(_IDP_SOURCE_POSITIONS)
                  & sleeper["player_id"].notna()].copy()
    if idp.empty:
        return pd.DataFrame(columns=["player_id", "name", "position", "team", "proj"])
    idp = idp.drop_duplicates("player_id")
    idp["position"] = idp["position"].replace(_IDP_NORMALIZE)
    idp["proj"] = _score_rows(idp, engine)
    keep = ["player_id", "name", "position", "team", "proj"]
    for adp_key in ("adp_idp", "adp_ppr"):
        if adp_key in idp.columns:
            keep.append(adp_key)
    return idp[keep].reset_index(drop=True)


def _hc_rows(league: LeagueSettings, season: int) -> pd.DataFrame:
    """HC board rows (synthetic ``HC:<team>`` ids) from win-probability EV."""
    ev = hc_draft_ev(league, season)
    if ev.empty:
        return pd.DataFrame(columns=["player_id", "name", "position", "team", "proj"])
    return pd.DataFrame({
        "player_id": ev["player_id"], "name": ev["coach_label"],
        "position": "HC", "team": ev["team"], "proj": ev["expected_season_points"],
    })


def _prior_season_offense(season: int, league: LeagueSettings) -> pd.DataFrame:
    """Fallback pseudo-projection: prior-season realized league points per player."""
    from fantasy.data.nfl import season_totals

    try:
        tot = season_totals([season - 1], ScoringEngine(league))
    except Exception as e:  # noqa: BLE001
        log.warning("Prior-season fallback totals unavailable (%s).", e)
        return pd.DataFrame(columns=["player_id", "name", "position", "team", "proj"])
    off = tot[~tot["position"].isin(IDP_POSITIONS)].copy()
    return pd.DataFrame({
        "player_id": off["player_id"], "name": off["player_display_name"],
        "position": off["position"], "team": np.nan, "proj": off["pts"],
    })


def build_season_board(
    season: int, league: LeagueSettings, refresh: bool = False
) -> pd.DataFrame:
    """Build the season value board for ``season`` under ``league``.

    Returns a frame sorted by VOR with at least: ``player_id, name, position,
    team, proj, return_pts, adp, adp_sd, vor, replacement, proj_source``.
    """
    engine = ScoringEngine(league)

    # ── (a) offense per-stat frames ──────────────────────────────────────────
    espn = _safe_espn(season, league, refresh)
    sleeper = _safe_sleeper(season, refresh)
    merged = _merge_offense(espn, sleeper)

    proj_source = "consensus"
    if merged.empty or (espn.empty and sleeper.empty):
        # ── (h) preseason fallback: prior-season realized points ─────────────
        merged = _prior_season_offense(season, league)
        proj_source = "prior_season"
        merged["proj"] = pd.to_numeric(merged.get("proj"), errors="coerce").fillna(0.0)
    else:
        # ── (b) score each offense row ───────────────────────────────────────
        merged = merged[merged["position"].notna()].copy()
        merged["proj"] = _score_rows(merged, engine)

    base_cols = ["player_id", "name", "position", "team", "proj"]
    board = merged[[c for c in base_cols if c in merged.columns]].copy()

    # ── (d) IDP rows (scored the same way) ───────────────────────────────────
    idp = _idp_rows(sleeper, engine)
    # ── (e) HC rows ──────────────────────────────────────────────────────────
    hc = _hc_rows(league, season)
    board = pd.concat([board, idp[base_cols] if not idp.empty else idp,
                       hc[base_cols] if not hc.empty else hc],
                      ignore_index=True)
    board = board.dropna(subset=["player_id"]).drop_duplicates("player_id").reset_index(drop=True)

    # Draft board = startable positions only. A position with no slot (e.g. D/ST
    # in this league) has a zero replacement rank, which would give its rows an
    # inflated VOR of proj-minus-nothing and pollute the round plan.
    counts = replacement_counts(league)
    startable = {p for p, n in counts.items() if n > 0}
    board = board[board["position"].map(pooled_position).isin(startable)].reset_index(drop=True)

    # ── (c) returner overlay (visible via return_pts) ────────────────────────
    overlay = return_points_overlay(league, season)
    board["return_pts"] = board["player_id"].map(overlay).fillna(0.0)
    board["proj"] = (pd.to_numeric(board["proj"], errors="coerce").fillna(0.0)
                     + board["return_pts"]).round(2)

    # ── (f) merge ADP (FFC offense/K + Sleeper positional ADP for IDP/HC) ─────
    board = _merge_adp(board, season, league, idp, sleeper)
    board["proj_source"] = proj_source

    # ── (g) cross-positional VOR ─────────────────────────────────────────────
    board = compute_vor(board, league)
    return board.reset_index(drop=True)


# ── source wrappers (never raise; empty on failure) ───────────────────────────
def _safe_espn(season: int, league: LeagueSettings, refresh: bool) -> pd.DataFrame:
    try:
        client = EspnClient(league_id=league.league_id or 1, season=season)
        return client.season_stat_projections(refresh=refresh)
    except Exception as e:  # noqa: BLE001
        log.warning("ESPN season projections unavailable (%s).", e)
        return pd.DataFrame(columns=["player_id", "name", "position", "team"])


def _safe_sleeper(season: int, refresh: bool) -> pd.DataFrame:
    try:
        return sleeper_season_projections(season, refresh=refresh)
    except Exception as e:  # noqa: BLE001
        log.warning("Sleeper season projections unavailable (%s).", e)
        return pd.DataFrame(columns=["player_id", "name", "position", "team"])


def _merge_adp(board: pd.DataFrame, season: int, league: LeagueSettings,
               idp: pd.DataFrame, sleeper: pd.DataFrame) -> pd.DataFrame:
    """Attach ``adp`` + ``adp_sd``. FFC covers offense/K; Sleeper positional ADP
    fills IDP; HC gets a static late-round ADP. Missing -> named defaults."""
    board["adp"] = np.nan
    board["adp_sd"] = np.nan
    try:
        ffc = load_ffc_adp(season, teams=league.team_count, fmt=_fmt_for_league(league))
        ffc = ffc.dropna(subset=["player_id"]).drop_duplicates("player_id")
        amap = dict(zip(ffc["player_id"], ffc["adp"]))
        smap = dict(zip(ffc["player_id"], ffc["sd"]))
        board["adp"] = board["player_id"].map(amap)
        board["adp_sd"] = board["player_id"].map(smap)
    except Exception as e:  # noqa: BLE001
        log.info("FFC ADP merge skipped (%s).", e)

    # Sleeper IDP ADP for IDP rows only (FFC carries no IDP; Sleeper uses 999.0
    # as its "undrafted" sentinel, which must fall through to the default).
    if not sleeper.empty and "adp_idp" in sleeper.columns:
        sadp = dict(zip(sleeper["player_id"], sleeper["adp_idp"]))
        need = board["adp"].isna() & board["position"].isin(_IDP_SOURCE_POSITIONS)
        board.loc[need, "adp"] = board.loc[need, "player_id"].map(sadp).apply(
            lambda v: max(float(v), _IDP_ADP_FLOOR)
            if pd.notna(v) and float(v) < 990 else np.nan)

    # HC static ADP.
    hc_mask = board["position"] == "HC"
    board.loc[hc_mask & board["adp"].isna(), "adp"] = _HC_ADP

    board["adp"] = board["adp"].fillna(_DEFAULT_ADP)
    board["adp_sd"] = board["adp_sd"].fillna(_DEFAULT_ADP_SD)
    return board
