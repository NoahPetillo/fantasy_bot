"""Transaction-feed moments: completed trades and notable FAAB pickups.

Source is ``client.recent_activity()`` → espn-api ``Activity`` objects, whose
``.actions`` is a list of ``(Team, action, Player|id, bid)`` tuples. Trades emit
paired ``TRADE_SENT``/``TRADE_RECEIVED`` rows (one per player); a waiver pickup is
a ``WAIVER ADDED`` row carrying the winning FAAB ``bid``.

Caveats baked in from the research + a live probe:
- The activity/``communication`` endpoint is a *current-season* feature; it 404s
  for past seasons. Callers must handle that (the activity cycle does).
- The trade record omits a tidy player list, so we reconstruct "who got what" by
  grouping the resolved ``TRADE_RECEIVED`` rows per team.
- These events aren't week-scoped, so ``week`` is fixed to 0 for stable
  idempotency and the card eyebrow shows the transaction DATE via ``period_label``.

Pure functions over duck-typed Activity objects — unit-testable without a league.
"""

from __future__ import annotations

import logging
from datetime import datetime

from fantasy.moments.models import Moment, MomentType

log = logging.getLogger(__name__)


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def _tname(team) -> str:
    return str(getattr(team, "team_name", None) or getattr(team, "team_abbrev", None) or "?")


def _pname(player) -> str:
    return str(getattr(player, "name", None) or player or "a player")


def _date_label(date_ms) -> str:
    try:
        return datetime.fromtimestamp(int(date_ms) / 1000).strftime("%b %d, %Y")
    except (TypeError, ValueError, OSError):
        return "Trade"


def _rows(activity):
    for tup in getattr(activity, "actions", []) or []:
        team, action, player, bid = (list(tup) + [None, None, None, 0])[:4]
        yield team, str(action or ""), player, bid


# ── completed trades ─────────────────────────────────────────────────────────
def detect_trades(activities: list, season: int) -> list[Moment]:
    out: list[Moment] = []
    for a in activities or []:
        date_ms = getattr(a, "date", 0) or 0
        received: dict[str, list[str]] = {}
        given: dict[str, list[str]] = {}
        for team, action, player, _bid in _rows(a):
            if action == "TRADE_RECEIVED" and team is not None:
                received.setdefault(_tname(team), []).append(_pname(player))
            elif action == "TRADE_SENT" and team is not None:
                given.setdefault(_tname(team), []).append(_pname(player))
        sides = received or given  # prefer the cleaner "who received what" view
        if not sides or not any(sides.values()):
            continue

        parts = [f"{tn} gets {', '.join(players)}" for tn, players in sides.items() if players]
        all_players = sorted({p for players in sides.values() for p in players})
        teams_in = list(sides.keys())
        out.append(Moment(
            type=MomentType.trade, season=season, week=0,
            period_label=_date_label(date_ms),
            spice=_clamp(78 + len(all_players) * 2, hi=96), team_id=None,
            dedup_key=f"trade:{int(date_ms)}:{'|'.join(all_players)}",
            headline=f"Trade alert: {' & '.join(teams_in)} make a deal",
            lines=parts,
            blurb="; ".join(parts) + ".",
        ))
    return out


# ── notable FAAB pickups ──────────────────────────────────────────────────────
def detect_waivers(activities: list, season: int, min_bid: int = 15) -> list[Moment]:
    out: list[Moment] = []
    for a in activities or []:
        date_ms = getattr(a, "date", 0) or 0
        for team, action, player, bid in _rows(a):
            if action != "WAIVER ADDED":
                continue
            bid = int(bid or 0)
            if bid < min_bid:
                continue
            pn, tn = _pname(player), _tname(team)
            out.append(Moment(
                type=MomentType.waiver, season=season, week=0,
                period_label=_date_label(date_ms),
                spice=_clamp(35 + min(50, bid * 0.9)), team_id=getattr(team, "team_id", None),
                dedup_key=f"waiver:{int(date_ms)}:{pn}:{bid}",
                big_stat=f"${bid}", player=pn, player_team=tn,
                headline=f"{tn} drops ${bid} FAAB on {pn}",
                blurb=f"{tn} won the bidding war for {pn} with a ${bid} FAAB bid.",
            ))
    return out
