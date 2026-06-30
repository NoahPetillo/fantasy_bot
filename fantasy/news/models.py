"""Typed news/injury signals — the structured output of the news layer.

Unstructured news and crowd/injury feeds are normalized into PlayerSignals keyed
by gsis ``player_id`` so they can adjust decisions (injury → start/sit alert;
trending add → waiver priority; depth change → projection nudge).
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class EventType(str, Enum):
    injury_out = "injury_out"
    injury_doubtful = "injury_doubtful"
    injury_questionable = "injury_questionable"
    injury_return = "injury_return"
    ir = "ir"
    trending_add = "trending_add"
    trending_drop = "trending_drop"
    depth_change = "depth_change"
    trade = "trade"
    news = "news"
    # Expert-signal events
    breakout = "breakout"
    usage_change = "usage_change"
    smoke_screen_warning = "smoke_screen_warning"
    buy_low = "buy_low"
    sell_high = "sell_high"


# Rough fantasy severity (how much it should move attention), 0..1.
SEVERITY = {
    EventType.injury_out: 0.9, EventType.ir: 0.95, EventType.injury_doubtful: 0.7,
    EventType.injury_questionable: 0.4, EventType.injury_return: 0.7,
    EventType.depth_change: 0.6, EventType.trade: 0.7,
    EventType.trending_add: 0.5, EventType.trending_drop: 0.3, EventType.news: 0.2,
    EventType.breakout: 0.6, EventType.usage_change: 0.55, EventType.buy_low: 0.4,
    EventType.sell_high: 0.4, EventType.smoke_screen_warning: 0.3,
}


class PlayerSignal(BaseModel):
    player_id: str | None = None  # gsis id (None if unmapped)
    player_name: str = ""
    position: str | None = None
    team: str | None = None
    event_type: EventType
    summary: str = ""
    source: str = ""
    source_url: str | None = None
    published_ts: str | None = None
    confidence: float = 0.6
    beneficiaries: list[str] = Field(default_factory=list)  # player names who gain

    @property
    def severity(self) -> float:
        return SEVERITY.get(self.event_type, 0.2) * self.confidence

    def key(self) -> str:
        return f"{self.player_id or self.player_name}:{self.event_type.value}:{self.published_ts or ''}"
