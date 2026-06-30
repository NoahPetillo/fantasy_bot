"""Moment — one noteworthy thing that happened in a fantasy week.

A Moment is a pure data record produced by the detector from box scores. It
carries everything two downstream steps need: the factual ``blurb`` (fed to the
caption writer) and the structured fields the graphic renderer draws (team names,
scores, the headline stat, a subject player). ``spice`` is the 0–100 ranking
score that decides which moments are worth posting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class MomentType(str, Enum):
    nailbiter = "nailbiter"          # decided by a hair (your "beat me by 1 point")
    blowout = "blowout"              # decided by a mile
    high_score = "high_score"        # week's top score
    low_score = "low_score"          # week's worst score
    bench_blunder = "bench_blunder"  # left the most points on the bench
    lucky = "lucky"                  # won despite a below-median score
    unlucky = "unlucky"              # lost despite an above-median score
    boom = "boom"                    # biggest starter overperformance vs projection
    bust = "bust"                    # biggest starter underperformance vs projection
    # ── Phase 2: standings + transactions ──
    hot_streak = "hot_streak"        # longest active win streak
    cold_streak = "cold_streak"      # longest active losing streak
    rivalry = "rivalry"              # configured rivalry pair just played
    trade = "trade"                  # a completed trade
    waiver = "waiver"                # a notable FAAB splash


# Whether a moment is about a single head-to-head matchup (gets a scoreboard
# layout) or a league-wide superlative / a single player.
MATCHUP_TYPES = {MomentType.nailbiter, MomentType.blowout, MomentType.lucky,
                 MomentType.unlucky, MomentType.rivalry}


@dataclass
class Moment:
    type: MomentType
    season: int
    week: int
    headline: str               # short punchy line: card title + proposal title
    blurb: str                  # factual 1–2 sentences; the caption writer's source
    spice: float = 0.0          # 0–100 ranking score (set by the detector)
    period_label: str | None = None  # card eyebrow override (e.g. a trade date); else "Week N"

    # Subject / dedup / proposal routing
    team_id: int | None = None  # the team the moment is "about" (proposal.team_id)
    manager: str | None = None  # first name of the subject team's manager (for roasting by name)
    dedup_key: str = ""         # stable identity within (kind, season, week)

    # Scoreboard fields (matchup moments)
    team_a: str | None = None
    team_b: str | None = None
    score_a: float | None = None
    score_b: float | None = None

    # Headline stat + optional subject player (bench/boom/bust, superlatives)
    big_stat: str | None = None      # e.g. "by 0.8", "184.6", "−27.4 vs proj"
    player: str | None = None
    player_team: str | None = None   # fantasy team that owns/started the player
    lines: list[str] | None = None   # mid-card body list (e.g. each side of a trade)

    extra: dict = field(default_factory=dict)

    def key_fields(self) -> dict:
        """Identity used for idempotency — one alert per moment per week."""
        return {"moment_type": self.type.value, "id": self.dedup_key}
