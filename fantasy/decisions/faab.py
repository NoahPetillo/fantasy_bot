"""FAAB bid sizing — a first-price sealed-bid auction (the genuine game-theory bit).

Bid scales with the marginal rest-of-season value of the pickup and saturates as a
fraction of the remaining budget, then is shaded down (you don't bid your full
valuation in a first-price auction). This is a principled v1; it's later refined
by learning each league's observed bid distribution from transaction history.
"""

from __future__ import annotations

import math


def suggest_bid(
    marginal_ros_value: float,
    budget: int,
    remaining_budget: int | None = None,
    scale: float = 55.0,
    shading: float = 0.85,
    min_bid: int = 1,
) -> int:
    """Return a whole-dollar FAAB bid for a pickup worth ``marginal_ros_value``
    points over what it replaces, given the league ``budget``.

    A marginal value ~= ``scale`` points spends ~63% of budget before shading.
    Non-positive value -> 0 (don't bid).
    """
    if marginal_ros_value <= 0:
        return 0
    cap = remaining_budget if remaining_budget is not None else budget
    frac = 1.0 - math.exp(-marginal_ros_value / scale)
    bid = int(round(budget * frac * shading))
    return max(min(bid, cap), min_bid if cap >= min_bid else 0)
