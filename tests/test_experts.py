"""Expert-signal pipeline: extraction, fusion + corroboration gate, bounded adjust."""

from __future__ import annotations

import fantasy.news.experts.extract as ex
from fantasy.news.experts.adjust import priority_boosts, projection_deltas
from fantasy.news.experts.fusion import fuse
from fantasy.news.experts.models import ExpertSignal, RawPost
from fantasy.news.models import EventType


def test_deterministic_extract_grounds_and_classifies(monkeypatch):
    monkeypatch.setattr(ex, "_name_index", lambda: {"christian mccaffrey": "00-0033280"})
    post = RawPost(id="1", author_handle="@x", outlet="ESPN", platform="bluesky",
                   text="Christian McCaffrey ruled out for Sunday.", base_trust=0.9)
    sigs = ex._deterministic(post)
    assert len(sigs) == 1
    assert sigs[0].player_id == "00-0033280"
    assert sigs[0].event_type == EventType.injury_out and sigs[0].direction == -1


def _sig(pid, outlet, direction=-1, etype=EventType.injury_out, base=0.9, **kw):
    return ExpertSignal(player_name="P", player_id=pid, event_type=etype, direction=direction,
                        outlet=outlet, base_trust=base, expert_handle="@" + outlet, **kw)


def test_two_independent_outlets_corroborate():
    fused = fuse([_sig("p1", "ESPN"), _sig("p1", "NFLNetwork")])
    assert len(fused) == 1
    assert fused[0].corroborated and fused[0].independent_outlets == 2


def test_single_low_trust_source_is_alert_only():
    fused = fuse([_sig("p1", "RandomBlog", base=0.6)])
    assert fused[0].corroborated is False


def test_single_tier1_breaker_injury_corroborates():
    # A high-trust breaker on an injury is actionable on its own.
    fused = fuse([_sig("p1", "ESPN", base=0.95)])
    assert fused[0].corroborated is True


def test_sarcasm_and_hypothetical_hard_zeroed():
    assert fuse([_sig("p1", "ESPN", is_sarcasm=True)]) == []
    assert fuse([_sig("p1", "ESPN", is_hypothetical=True)]) == []


def test_projection_delta_and_boost_are_capped():
    fused = fuse([_sig("p1", "ESPN"), _sig("p1", "NFLNetwork")])  # corroborated injury_out
    deltas = projection_deltas(fused, {"p1": 100.0})
    assert "p1" in deltas and -15.0 <= deltas["p1"] <= 0.0  # negative, within 15% cap

    add = fuse([_sig("p2", "ESPN", direction=1, etype=EventType.breakout),
                _sig("p2", "Underdog", direction=1, etype=EventType.breakout)])
    boosts = priority_boosts(add)
    assert boosts and all(1.0 <= b <= 1.5 for b in boosts.values())


def test_uncorroborated_does_not_adjust():
    fused = fuse([_sig("p3", "RandomBlog", base=0.6, etype=EventType.usage_change, direction=1)])
    assert projection_deltas(fused, {"p3": 100.0}) == {}
