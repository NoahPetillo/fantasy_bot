"""ProjectionService — the trained model as a service.

Fits the per-position model + stacking blender + variance model once, then
produces a current-week value board (projection, floor/median/ceiling, VOR) for
any (season, week). This is what the start/sit, waiver, and trade generators all
read from, so they share one consistent, league-adaptive valuation.
"""

from __future__ import annotations

import logging

import pandas as pd

from fantasy.data.nfl import load_weekly
from fantasy.league_settings import LeagueSettings
from fantasy.projections.distributions import VarianceModel
from fantasy.projections.ensemble import StackedBlender
from fantasy.projections.features import build_features
from fantasy.projections.model import ProjectionModel
from fantasy.valuation.scoring import ScoringEngine
from fantasy.valuation.vor import compute_vor

log = logging.getLogger(__name__)


def overlay_espn(cur: pd.DataFrame, espn_proj: dict[str, float] | None) -> pd.DataFrame:
    """Set ``proj`` to ESPN's number where available (primary), else keep our
    model's, and record ``proj_source`` per row."""
    if espn_proj:
        s = cur["player_id"].map(espn_proj)
        cur["proj_source"] = s.notna().map({True: "espn", False: "model"})
        cur["proj"] = s.fillna(cur["proj"])
    else:
        cur["proj_source"] = "model"
    return cur


class ProjectionService:
    def __init__(self, league: LeagueSettings):
        self.league = league
        self.engine = ScoringEngine(league)
        self.train_seasons: list[int] = []
        self.model: ProjectionModel | None = None
        self.blender = StackedBlender()
        self.varmodel = VarianceModel()

    def fit(self, train_seasons: list[int]) -> "ProjectionService":
        self.train_seasons = sorted(train_seasons)
        feat = build_features(load_weekly(self.train_seasons), self.engine)
        self.model = ProjectionModel().fit(feat)

        seasons = sorted(feat["season"].unique())
        if len(seasons) >= 2:
            fit_s, val_s = seasons[:-1], seasons[-1]
            tmp = ProjectionModel().fit(feat[feat["season"].isin(fit_s)])
            val = feat[feat["season"] == val_s].copy()
            val["proj"] = tmp.predict(val)
            val = val[val["games_so_far"] >= 1]
            self.blender.fit(val)
            self.varmodel.fit(val)  # out-of-sample residuals
        else:
            scored = feat.assign(proj=self.model.predict(feat))
            self.varmodel.fit(scored)
        log.info("ProjectionService fit on %s", self.train_seasons)
        return self

    def total_weeks(self) -> int:
        return self.league.regular_season_weeks + len(self.league.playoff_weeks)

    def remaining_weeks(self, week: int) -> int:
        return max(self.total_weeks() - week + 1, 1)

    def project(self, season: int, week: int, weekly: pd.DataFrame | None = None,
                espn_proj: dict[str, float] | None = None,
                fused_signals: list | None = None) -> pd.DataFrame:
        """Weekly value board for (season, week): proj, floor/median/ceiling, VOR.

        If ``espn_proj`` (gsis_id -> ESPN projected points) is supplied, ESPN's
        number is used as the primary projection (it edges our model ~2%); our
        model fills any gaps and always supplies the distribution + VOR.

        If ``fused_signals`` (corroborated expert signals) is supplied AND
        EXPERT_ADJUST_DECISIONS is on, a capped (±15%) projection nudge is applied
        before VOR/distributions recompute — so injuries/usage news move the board.
        """
        assert self.model is not None, "call fit() first"
        seasons = sorted(set(self.train_seasons) | {season})
        wk = weekly if weekly is not None else load_weekly(seasons)
        feat = build_features(wk, self.engine)
        cur = feat[(feat["season"] == season) & (feat["week"] == week)].copy()
        if cur.empty:
            return cur
        cur["proj_model"] = self.model.predict(cur)
        cur["proj"] = self.blender.predict(cur.assign(proj=cur["proj_model"])) \
            if self.blender.global_weights is not None else cur["proj_model"]

        from fantasy.config import settings
        if settings.projection_consensus:
            cur = self._consensus(cur, season, week, espn_proj)
        else:
            cur = overlay_espn(cur, espn_proj)
        cur = self._apply_expert_deltas(cur, fused_signals)
        cur = self._apply_usage(cur, season, week, fused_signals)
        return self._finish_board(cur)

    def _apply_usage(self, cur: pd.DataFrame, season: int, week: int,
                     fused_signals: list | None) -> pd.DataFrame:
        """Boost the next man up when a corroborated injury rules out a starter."""
        from fantasy.config import settings

        if not (fused_signals and settings.expert_adjust_decisions):
            return cur
        from fantasy.news.models import EventType
        from fantasy.projections.usage import vacated_boosts

        out_ids = [f.player_id for f in fused_signals
                   if getattr(f, "corroborated", False)
                   and f.event_type in (EventType.injury_out, EventType.ir)]
        boosts = vacated_boosts(out_ids, cur)
        if boosts:
            cur["proj"] = (cur["proj"] + cur["player_id"].map(boosts).fillna(0.0)).round(2)
        return cur

    def _consensus(self, cur: pd.DataFrame, season: int, week: int,
                   espn_proj: dict[str, float] | None) -> pd.DataFrame:
        """Average our model with ESPN + Sleeper (whatever's available) per player —
        the wisdom-of-crowds backbone. Falls back to our model where uncovered."""
        from fantasy.projections.consensus import consensus
        from fantasy.projections.props import PlayerPropSource
        from fantasy.projections.sources import SleeperProjectionSource

        sources: dict[str, dict] = {"model": dict(zip(cur["player_id"], cur["proj"]))}
        if espn_proj:
            sources["espn"] = espn_proj
        sl = SleeperProjectionSource().weekly_points(season, week, self.league)
        if sl:
            sources["sleeper"] = sl
        if PlayerPropSource.enabled():  # the sharpest source, if an odds key is set
            pr = PlayerPropSource().weekly_points(season, week, self.league)
            if pr:
                sources["props"] = pr
        means, counts = consensus(sources)
        cur["proj_sources"] = cur["player_id"].map(counts).fillna(1).astype(int)
        cur["proj_source"] = "+".join(sources.keys())
        cur["proj"] = cur["player_id"].map(means).fillna(cur["proj"]).round(2)
        return cur

    def _apply_expert_deltas(self, cur: pd.DataFrame, fused_signals: list | None) -> pd.DataFrame:
        from fantasy.config import settings

        cur["expert_delta"] = 0.0
        if not (fused_signals and settings.expert_adjust_decisions):
            return cur
        from fantasy.news.experts.adjust import projection_deltas

        proj_map = dict(zip(cur["player_id"], cur["proj"]))
        deltas = projection_deltas(fused_signals, proj_map)  # capped ±15% inside
        if deltas:
            cur["expert_delta"] = cur["player_id"].map(deltas).fillna(0.0).round(2)
            cur["proj"] = (cur["proj"] + cur["expert_delta"]).clip(lower=0.0)
        return cur

    def _finish_board(self, cur: pd.DataFrame) -> pd.DataFrame:
        """Attach distributions (from our variance model, scaled to the final mean)
        and compute VOR, then sort into a value board."""
        q = [self.varmodel.quantiles(p, m) for p, m in zip(cur["position"], cur["proj"])]
        cur["floor"] = [round(x[0.1], 1) for x in q]
        cur["median"] = [round(x[0.5], 1) for x in q]
        cur["ceiling"] = [round(x[0.9], 1) for x in q]
        cur["sd"] = [round(self.varmodel.sd(p, m), 2) for p, m in zip(cur["position"], cur["proj"])]

        board = compute_vor(
            cur[["player_id", "player_display_name", "position", "team", "opponent_team",
                 "proj", "proj_source", "floor", "median", "ceiling", "sd"]],
            self.league,
        )
        board["proj"] = board["proj"].round(2)
        return board.reset_index(drop=True)
