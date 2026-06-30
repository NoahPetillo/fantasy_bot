"""Data shapes for the expert-signal pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field

from fantasy.news.models import EventType


class RawPost(BaseModel):
    """A normalized post from any source (Bluesky / RSS / X)."""

    id: str
    author_handle: str          # the registry key this came from
    outlet: str                 # for independence counting
    platform: str               # bluesky | rss | x
    text: str
    ts: str | None = None       # ISO timestamp
    url: str | None = None
    base_trust: float = 0.6


class ExpertSignal(BaseModel):
    """Structured extraction from one post about one player."""

    player_name: str
    player_id: str | None = None        # gsis (None until resolved / dropped if unresolved)
    event_type: EventType
    direction: int = 0                  # +1 helps, -1 hurts, 0 neutral
    confidence: float = 0.5
    is_sarcasm: bool = False
    is_hedge: bool = False
    is_hypothetical: bool = False
    expert_handle: str = ""
    outlet: str = ""
    base_trust: float = 0.6
    ts: str | None = None
    source_url: str | None = None


class FusedSignal(BaseModel):
    """Trust-weighted consensus for one (player, event family)."""

    player_id: str
    player_name: str
    event_type: EventType
    direction: int
    fused_confidence: float
    trust_weight: float
    independent_outlets: int
    corroborated: bool
    experts: list[str] = Field(default_factory=list)
    rationale: str = ""
