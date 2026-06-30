"""Demo the expert-signal layer against FREE live sources (Bluesky + RSS).

    uv run python scripts/expert_signals_demo.py

Pulls recent posts from the curated registry's Bluesky handles + RSS feeds,
extracts player signals, fuses them with trust + the corroboration gate, and
prints what WOULD move a decision vs. what stays alert-only. (Offseason now, so
expect thin signal — this proves the pipeline, not in-season volume.)
"""

from __future__ import annotations

import logging

from fantasy.news.experts import gather_posts, load_registry
from fantasy.news.experts.adjust import alerts, priority_boosts
from fantasy.news.experts.extract import extract_signals
from fantasy.news.experts.fusion import fuse

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


def main() -> int:
    reg = load_registry()
    print(f"Registry: {len(reg.experts)} experts "
          f"({len(reg.with_bluesky())} on Bluesky), {len(reg.rss_feeds)} RSS feeds.\n")

    posts = gather_posts(reg)
    print(f"Pulled {len(posts)} posts from free sources.")
    signals = extract_signals(posts)
    print(f"Extracted {len(signals)} player signals (grounded to gsis ids).")
    fused = fuse(signals)

    actionable = [f for f in fused if f.corroborated]
    boosts = priority_boosts(fused)
    print(f"\nCorroborated (would adjust decisions): {len(actionable)}")
    for f in actionable[:10]:
        b = boosts.get(f.player_id)
        print(f"  • {f.player_name:22s} {f.event_type.value:18s} dir={f.direction:+d} "
              f"conf={f.fused_confidence:.2f} outlets={f.independent_outlets}"
              + (f"  waiver×{b}" if b else ""))

    al = alerts(fused)
    print(f"\nAlert-only (single-source, surfaced to you, never auto-applied): {len(al)}")
    for f in al[:10]:
        print(f"  • {f.player_name:22s} {f.event_type.value:18s} "
              f"conf={f.fused_confidence:.2f} via {', '.join(f.experts[:2])}")

    _inseason_example()
    print("\n✓ Expert-signal pipeline works end-to-end on free sources "
          "(Bluesky + RSS), gated and trust-weighted.")
    return 0


def _inseason_example() -> None:
    """Run realistic in-season posts through the SAME pipeline to show the output."""
    from fantasy.news.experts.models import RawPost

    posts = [
        RawPost(id="1", author_handle="@AdamSchefter", outlet="ESPN", platform="x",
                text="Sources: Christian McCaffrey has been ruled out for Sunday.", base_trust=0.95),
        RawPost(id="2", author_handle="@RapSheet", outlet="NFLNetwork", platform="x",
                text="Christian McCaffrey will not play Sunday, per sources.", base_trust=0.95),
        RawPost(id="3", author_handle="@FieldYates", outlet="ESPN", platform="x",
                text="With McCaffrey out, Jordan Mason is the must-add waiver priority this week.",
                base_trust=0.88),
        RawPost(id="4", author_handle="@HaydenWinks", outlet="Underdog", platform="x",
                text="Jordan Mason set for a workhorse role and huge snap share — league winner add.",
                base_trust=0.82),
        RawPost(id="5", author_handle="@SomeBlog", outlet="RandomBlog", platform="x",
                text="Heard a whisper that Rashee Rice might be a sneaky buy low, who knows lol.",
                base_trust=0.55),
    ]
    fused = fuse(extract_signals(posts))
    boosts = priority_boosts(fused)
    print("\n──────── ILLUSTRATIVE in-season example (synthetic posts, real pipeline) ────────")
    for f in fused:
        tag = "ADJUSTS DECISION" if f.corroborated else "alert-only"
        b = f"  waiver×{boosts[f.player_id]}" if f.player_id in boosts else ""
        print(f"  [{tag:16s}] {f.player_name:20s} {f.event_type.value:14s} dir={f.direction:+d} "
              f"conf={f.fused_confidence:.2f} outlets={f.independent_outlets}{b}")
        print(f"       {f.rationale}; experts: {', '.join(f.experts)}")


if __name__ == "__main__":
    raise SystemExit(main())
