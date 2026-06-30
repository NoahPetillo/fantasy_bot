"""League content engine — turn a finished fantasy week into shareable hype.

A *secondary* feature that sits on top of the read-only ESPN layer and reuses the
notify-and-approve loop: scan the league for exciting moments (nail-biters,
blowouts, bench blunders, lucky/unlucky results, boom/bust performances), rank
them by "spiciness", generate a caption + a square graphic for the top few, and
raise them as ``moment`` proposals. On approval the moment posts to the league
Discord channel. The same graphic is saved to disk so it can be hand-posted to
Instagram — IG is intentionally NOT automated.

Nothing here writes to ESPN; "executing" a moment just posts a message.
"""

from __future__ import annotations

from fantasy.moments.activity import detect_trades, detect_waivers
from fantasy.moments.cycle import activity_cycle, content_cycle
from fantasy.moments.detector import detect_moments
from fantasy.moments.models import Moment, MomentType
from fantasy.moments.score import rank_and_select, spiciness
from fantasy.moments.standings import detect_rivalries, detect_streaks

__all__ = [
    "Moment",
    "MomentType",
    "activity_cycle",
    "content_cycle",
    "detect_moments",
    "detect_rivalries",
    "detect_streaks",
    "detect_trades",
    "detect_waivers",
    "rank_and_select",
    "spiciness",
]
