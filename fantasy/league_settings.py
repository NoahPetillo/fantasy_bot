"""LeagueSettings — the league's full configuration as a first-class object.

This is the backbone of league-adaptivity. EVERY valuation flows from here:

- scoring map (statId/canonical -> points) -> :class:`fantasy.valuation.scoring.ScoringEngine`
- roster slots + team count          -> VOR replacement baselines, lineup LP
- waiver type + FAAB budget          -> FAAB auction bidder
- playoff weeks                      -> trade generator's playoff-weighted gain
- keeper/dynasty flags               -> draft/trade horizon (redraft = ROS only)

Nothing downstream hardcodes "PPR" or "12 teams" or "1 QB". They read it from an
instance of this class, which is populated live from ESPN ``mSettings`` (see
:func:`fantasy.espn.client.EspnClient.league_settings`). The same object is what
backtests load from a stored season so historical scoring matches the real league.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from fantasy.espn.stat_ids import FLEX_ELIGIBILITY, STARTER_SLOTS


class ScoringFormat(str, Enum):
    standard = "standard"
    half_ppr = "half_ppr"
    ppr = "ppr"
    custom = "custom"


class WaiverType(str, Enum):
    faab = "faab"  # free-agent acquisition budget (sealed bid)
    rolling = "rolling"  # rolling priority list
    reverse = "reverse"  # reverse standings order
    none = "none"


class RosterRequirements(BaseModel):
    """Count of each lineup slot, e.g. {'QB':1,'RB':2,'WR':2,'TE':1,'FLEX':1,...}."""

    slots: dict[str, int] = Field(default_factory=dict)

    @property
    def starter_slots(self) -> dict[str, int]:
        return {s: n for s, n in self.slots.items() if s in STARTER_SLOTS and n > 0}

    @property
    def total_starters(self) -> int:
        return sum(self.starter_slots.values())

    @property
    def bench_size(self) -> int:
        return self.slots.get("BE", 0)

    @property
    def has_superflex(self) -> bool:
        return self.slots.get("OP", 0) > 0 or self.slots.get("TQB", 0) > 0

    def starters_at_position(self, position: str) -> float:
        """Expected starting roster spots a position fills league-wide per team,
        counting dedicated slots plus its fractional share of each flex it's
        eligible for. Drives the VOR replacement baseline (last starter rank)."""
        dedicated = self.starter_slots.get(position, 0)
        flex_share = 0.0
        for slot, count in self.starter_slots.items():
            elig = FLEX_ELIGIBILITY.get(slot)
            if elig and position in elig:
                flex_share += count / len(elig)
        return dedicated + flex_share


class LeagueSettings(BaseModel):
    """Everything about the league that changes how players are valued."""

    league_id: int | None = None
    season: int | None = None
    name: str | None = None

    team_count: int = 12
    roster: RosterRequirements = Field(default_factory=RosterRequirements)

    # Scoring: canonical-stat-name -> points-per-unit (the source of truth).
    scoring: dict[str, float] = Field(default_factory=dict)
    # Raw ESPN statId -> points, kept verbatim for audit / Phase-0 verification.
    scoring_items_raw: dict[int, float] = Field(default_factory=dict)
    # Position-specific reception bonus (TE-premium), e.g. {'TE': 0.5}.
    position_reception_bonus: dict[str, float] = Field(default_factory=dict)

    # Waivers / acquisitions.
    waiver_type: WaiverType = WaiverType.faab
    faab_budget: int = 100
    acquisition_limit: int | None = None  # season-long add limit, if any

    # Schedule / playoffs (drives playoff-weighted trade value).
    regular_season_weeks: int = 14
    playoff_team_count: int = 6
    playoff_weeks: list[int] = Field(default_factory=lambda: [15, 16, 17])
    matchup_periods: int | None = None

    # Format flags.
    keeper_count: int = 0
    is_dynasty: bool = False
    uses_idp: bool = False

    # ── derived helpers used across the codebase ──
    @property
    def scoring_format(self) -> ScoringFormat:
        """Best-effort classification purely for display / FantasyCalc params.
        Internal math always uses the full `scoring` map, never this label."""
        rec = self.scoring.get("receptions", 0.0)
        if abs(rec) < 1e-9:
            return ScoringFormat.standard
        if abs(rec - 0.5) < 1e-6:
            return ScoringFormat.half_ppr
        if abs(rec - 1.0) < 1e-6:
            return ScoringFormat.ppr
        return ScoringFormat.custom

    @property
    def ppr_value(self) -> float:
        """Points-per-reception as a float (for FantasyCalc `ppr=` param, etc.)."""
        return float(self.scoring.get("receptions", 0.0))

    @property
    def num_qbs_effective(self) -> int:
        """1 for standard, 2 for superflex/2QB — used by FantasyCalc trade values."""
        return 2 if self.roster.has_superflex else 1

    @property
    def offensive_positions(self) -> list[str]:
        return ["QB", "RB", "WR", "TE", "K", "D/ST"]

    def summary(self) -> str:
        r = ", ".join(f"{k}:{v}" for k, v in self.roster.starter_slots.items())
        return (
            f"{self.name or 'League'} ({self.season}) — {self.team_count} teams, "
            f"{self.scoring_format.value} (rec={self.ppr_value}), "
            f"starters[{r}], bench {self.roster.bench_size}, "
            f"waivers {self.waiver_type.value}"
            + (f" (FAAB ${self.faab_budget})" if self.waiver_type == WaiverType.faab else "")
            + f", playoffs wks {self.playoff_weeks}"
            + (" [SUPERFLEX]" if self.roster.has_superflex else "")
            + (" [IDP]" if self.uses_idp else "")
            + (" [DYNASTY]" if self.is_dynasty else (f" [KEEP {self.keeper_count}]" if self.keeper_count else ""))
        )
