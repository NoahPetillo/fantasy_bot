"""Plan tiers + limits. Tune the numbers here — everything else reads these.

Pro unlocks a higher daily chat quota and more connected leagues (the two things
that cost real money/compute per user). ``users.plan`` holds the effective tier.
"""

from __future__ import annotations

FREE = "free"
PRO = "pro"

PLANS: dict[str, dict] = {
    FREE: {"label": "Free", "chat_daily": 25, "max_leagues": 1},
    PRO: {"label": "Pro", "chat_daily": 1000, "max_leagues": 10},
}


def _plan(plan: str | None) -> dict:
    return PLANS.get((plan or FREE), PLANS[FREE])


def label(plan: str | None) -> str:
    return _plan(plan)["label"]


def chat_daily_limit(plan: str | None) -> int:
    return _plan(plan)["chat_daily"]


def max_leagues(plan: str | None) -> int:
    return _plan(plan)["max_leagues"]
