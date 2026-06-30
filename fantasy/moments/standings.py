"""Standings-derived moments: win/loss streaks and rivalry results.

Both come from ``client.teams()`` (espn-api ``Team`` objects), not box scores:

- **Streaks** are handed to us directly by ESPN — ``Team.streak_type`` ("WIN"/"LOSS")
  and ``Team.streak_length`` already reflect games through the latest decided week,
  so we don't recompute them.
- **Rivalries** use ``Team.schedule`` (opponent per matchup period), ``Team.outcomes``
  ("W"/"L"/"T"/"U") and ``Team.scores`` to detect that a configured pair just played
  this week and to tally the head-to-head series.

Pure functions over duck-typed teams, so they unit-test without a live league.
"""

from __future__ import annotations

import logging

from fantasy.moments.models import Moment, MomentType

log = logging.getLogger(__name__)

_DECIDED = {"W", "L", "T"}


def _name(t) -> str:
    return str(getattr(t, "team_name", None) or getattr(t, "team_abbrev", None) or "?")


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


# ── streaks ──────────────────────────────────────────────────────────────────
def detect_streaks(teams: list, season: int, week: int, min_len: int = 3) -> list[Moment]:
    """Flag the longest active win streak and the longest active losing streak."""
    hot = cold = None  # (length, team)
    for t in teams:
        stype = str(getattr(t, "streak_type", "") or "").upper()
        slen = int(getattr(t, "streak_length", 0) or 0)
        if slen < min_len:
            continue
        if stype == "WIN" and (hot is None or slen > hot[0]):
            hot = (slen, t)
        elif stype == "LOSS" and (cold is None or slen > cold[0]):
            cold = (slen, t)

    out: list[Moment] = []
    if hot is not None:
        n, t = hot
        out.append(Moment(
            type=MomentType.hot_streak, season=season, week=week,
            spice=_clamp(46 + (n - min_len) * 7), team_id=getattr(t, "team_id", None),
            dedup_key=f"hot:{getattr(t,'team_id','?')}:{n}",
            team_a=_name(t), big_stat=f"{n} STRAIGHT", player=None,
            headline=f"{_name(t)} is rolling — {n} wins in a row",
            blurb=f"{_name(t)} has won {n} straight heading into Week {week}. Somebody stop them.",
        ))
    if cold is not None:
        n, t = cold
        out.append(Moment(
            type=MomentType.cold_streak, season=season, week=week,
            spice=_clamp(44 + (n - min_len) * 7), team_id=getattr(t, "team_id", None),
            dedup_key=f"cold:{getattr(t,'team_id','?')}:{n}",
            team_a=_name(t), big_stat=f"{n} STRAIGHT", player=None,
            headline=f"{_name(t)} is in free fall — {n} losses in a row",
            blurb=f"{_name(t)} has dropped {n} straight going into Week {week}. It's getting ugly.",
        ))
    return out


# ── rivalries ────────────────────────────────────────────────────────────────
def _opponent_id(sched_entry) -> int | None:
    """A schedule entry may be an opponent Team object or a raw team id."""
    if sched_entry is None:
        return None
    return getattr(sched_entry, "team_id", sched_entry)


def _resolve_team(token: str, teams: list):
    """Match a rivalry token (team id, name, or abbrev) to a Team."""
    tok = str(token).strip().lower()
    if tok.isdigit():
        tid = int(tok)
        return next((t for t in teams if getattr(t, "team_id", None) == tid), None)
    for t in teams:
        if tok in (_name(t).lower(), str(getattr(t, "team_abbrev", "")).lower()):
            return t
    # loose contains-match as a fallback
    return next((t for t in teams if tok in _name(t).lower()), None)


def detect_rivalries(teams: list, season: int, week: int,
                     pairs: list[list[str]] | None) -> list[Moment]:
    """Emit a moment for each configured rivalry pair that PLAYED in ``week``,
    carrying the running head-to-head series record."""
    if not pairs:
        return []
    idx = week - 1  # outcomes/schedule are 0-based by matchup period
    out: list[Moment] = []
    for pair in pairs:
        if not pair or len(pair) < 2:
            continue
        a = _resolve_team(pair[0], teams)
        b = _resolve_team(pair[1], teams)
        if a is None or b is None or a is b:
            log.info("Rivalry pair %s did not resolve to two teams; skipping.", pair)
            continue
        a_sched = list(getattr(a, "schedule", []) or [])
        a_out = list(getattr(a, "outcomes", []) or [])
        a_scores = list(getattr(a, "scores", []) or [])
        # Did they play THIS week, and is it decided?
        if idx < 0 or idx >= len(a_sched) or idx >= len(a_out):
            continue
        if _opponent_id(a_sched[idx]) != getattr(b, "team_id", None):
            continue
        if a_out[idx] not in _DECIDED:
            continue

        # Running series across all decided meetings through this week.
        a_wins = b_wins = 0
        for j in range(min(idx + 1, len(a_sched), len(a_out))):
            if _opponent_id(a_sched[j]) == getattr(b, "team_id", None) and a_out[j] in _DECIDED:
                if a_out[j] == "W":
                    a_wins += 1
                elif a_out[j] == "L":
                    b_wins += 1
        result = a_out[idx]
        winner, loser = (a, b) if result == "W" else (b, a)
        a_pts = a_scores[idx] if idx < len(a_scores) and a_scores[idx] is not None else None
        # b's score for this week = b.scores[idx] (same period)
        b_scores = list(getattr(b, "scores", []) or [])
        b_pts = b_scores[idx] if idx < len(b_scores) and b_scores[idx] is not None else None
        wpts, lpts = (a_pts, b_pts) if winner is a else (b_pts, a_pts)
        series = (f"now lead the series {max(a_wins, b_wins)}-{min(a_wins, b_wins)}"
                  if a_wins != b_wins else f"even the series {a_wins}-{b_wins}")
        out.append(Moment(
            type=MomentType.rivalry, season=season, week=week,
            spice=_clamp(58 + (4 if a_wins == b_wins else 0)),
            team_id=getattr(loser, "team_id", None),
            dedup_key=f"riv:{'-'.join(sorted([str(getattr(a,'team_id','')), str(getattr(b,'team_id',''))]))}:{week}",
            team_a=_name(winner), score_a=wpts, team_b=_name(loser), score_b=lpts,
            big_stat=f"{max(a_wins,b_wins)}-{min(a_wins,b_wins)}",
            headline=f"Rivalry Week: {_name(winner)} takes down {_name(loser)}",
            blurb=(f"In their Week {week} rivalry clash, {_name(winner)} beat {_name(loser)}"
                   + (f" {wpts:.1f}–{lpts:.1f}" if wpts is not None and lpts is not None else "")
                   + f". {_name(winner)} {series}."),
        ))
    return out
