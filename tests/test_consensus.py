"""Multi-source consensus: equal-weight mean, per-player source counts, min_sources."""

from __future__ import annotations

from fantasy.projections.consensus import consensus


def test_equal_weight_mean_and_counts():
    means, counts = consensus({
        "model": {"p1": 10.0, "p2": 20.0},
        "espn": {"p1": 14.0, "p3": 9.0},
        "sleeper": {"p1": 12.0},
    })
    assert means["p1"] == 12.0 and counts["p1"] == 3   # mean(10,14,12)
    assert means["p2"] == 20.0 and counts["p2"] == 1
    assert means["p3"] == 9.0 and counts["p3"] == 1


def test_min_sources_filter():
    means, _ = consensus(
        {"a": {"p1": 10.0, "p2": 5.0}, "b": {"p1": 20.0}}, min_sources=2)
    assert "p1" in means and means["p1"] == 15.0
    assert "p2" not in means  # only one source


def test_empty_sources():
    means, counts = consensus({})
    assert means == {} and counts == {}
