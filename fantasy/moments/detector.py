"""Detect noteworthy moments from a week's ESPN box scores.

``detect_moments`` is a pure function over a list of espn-api ``BoxScore`` objects
(or any duck-typed stand-ins with the same attributes), so it's fully unit-testable
without a live league. Each detector emits zero or more :class:`Moment` records
with a 0–100 ``spice`` already attached.

What's reliable here: final scores, margins, per-player actual + projected points,
and starter-vs-bench slots — all confirmed present on espn-api's BoxPlayer. What's
deliberately NOT attempted: "who scored the literal last point" (no play-by-play
is available) — the nail-biter margin already captures the come-down-to-the-wire
feeling without pretending to know game-clock timing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from statistics import median

from fantasy.moments.config import content_config as settings  # decoupled from the app
from fantasy.moments.models import Moment, MomentType

log = logging.getLogger(__name__)

_BENCH_SLOTS = {"BE", "IR"}


@dataclass
class _Side:
    team_id: int | None
    name: str
    score: float
    lineup: list


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def _f(v) -> float:
    try:
        return float(v or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _sides(box) -> list[_Side]:
    out: list[_Side] = []
    for prefix in ("home", "away"):
        team = getattr(box, f"{prefix}_team", None)
        if team is None:  # BYE side
            continue
        name = getattr(team, "team_name", None) or getattr(team, "team_abbrev", None) or "?"
        out.append(_Side(
            team_id=getattr(team, "team_id", None),
            name=str(name),
            score=_f(getattr(box, f"{prefix}_score", 0.0)),
            lineup=list(getattr(box, f"{prefix}_lineup", []) or []),
        ))
    return out


def _starters(side: _Side) -> list:
    return [p for p in side.lineup if getattr(p, "slot_position", None) not in _BENCH_SLOTS]


def _bench(side: _Side) -> list:
    return [p for p in side.lineup if getattr(p, "slot_position", None) == "BE"]


def _matchup_key(a: _Side, b: _Side) -> str:
    ids = sorted(str(s.team_id) for s in (a, b))
    return f"{ids[0]}-{ids[1]}"


def detect_moments(box_scores: list, season: int, week: int) -> list[Moment]:
    """Return every detected moment for the week, each with spice scored."""
    sides_all: list[_Side] = []
    matchups: list[tuple[_Side, _Side]] = []
    for box in box_scores:
        sides = _sides(box)
        sides_all.extend(sides)
        if len(sides) == 2:
            matchups.append((sides[0], sides[1]))

    # A week with no real scores yet (all zeros) isn't worth recapping.
    if not sides_all or all(s.score <= 0 for s in sides_all):
        log.info("No scored games for %s wk%s — no moments.", season, week)
        return []

    league_median = median(s.score for s in sides_all)
    moments: list[Moment] = []
    moments += _close_and_blowout(matchups, season, week)
    moments += _luck(matchups, league_median, season, week)
    moments += _superlatives(sides_all, season, week)
    moments += _bench_blunder(sides_all, season, week)
    moments += _boom_bust(sides_all, season, week)
    return moments


# ── margin-based: nail-biter / blowout ──────────────────────────────────────
def _close_and_blowout(matchups, season, week) -> list[Moment]:
    out: list[Moment] = []
    for a, b in matchups:
        margin = abs(a.score - b.score)
        if margin <= 0:
            continue  # exact tie — skip (winner/loser undefined; vanishingly rare)
        winner, loser = (a, b) if a.score > b.score else (b, a)
        key = _matchup_key(a, b)
        if margin < settings.content_nailbiter_margin:
            spice = _clamp(72 + 28 * (1 - margin / settings.content_nailbiter_margin))
            out.append(Moment(
                type=MomentType.nailbiter, season=season, week=week, spice=spice,
                team_id=loser.team_id, dedup_key=key,
                team_a=winner.name, score_a=winner.score, team_b=loser.name, score_b=loser.score,
                big_stat=f"by {margin:.1f}",
                headline=f"{winner.name} survives {loser.name} by {margin:.1f}",
                blurb=(f"{winner.name} beat {loser.name} by just {margin:.1f} points "
                       f"({winner.score:.1f}–{loser.score:.1f}) in Week {week} — a true nail-biter."),
            ))
        elif margin > settings.content_blowout_margin:
            spice = _clamp(58 + min(40, margin - settings.content_blowout_margin))
            out.append(Moment(
                type=MomentType.blowout, season=season, week=week, spice=spice,
                team_id=loser.team_id, dedup_key=key,
                team_a=winner.name, score_a=winner.score, team_b=loser.name, score_b=loser.score,
                big_stat=f"by {margin:.1f}",
                headline=f"{winner.name} demolishes {loser.name} by {margin:.1f}",
                blurb=(f"{winner.name} blew out {loser.name} {winner.score:.1f}–{loser.score:.1f} "
                       f"in Week {week}, a {margin:.1f}-point beatdown."),
            ))
    return out


# ── luck vs the league median ────────────────────────────────────────────────
def _luck(matchups, league_median, season, week) -> list[Moment]:
    lucky: tuple[float, Moment] | None = None
    unlucky: tuple[float, Moment] | None = None
    for a, b in matchups:
        if a.score == b.score:
            continue
        winner, loser = (a, b) if a.score > b.score else (b, a)
        # Unluckiest: lost with a score above the league median (high-scoring loser).
        gap_loser = loser.score - league_median
        if gap_loser > 0:
            spice = _clamp(50 + min(42, gap_loser * 1.5))
            m = Moment(
                type=MomentType.unlucky, season=season, week=week, spice=spice,
                team_id=loser.team_id, dedup_key=str(loser.team_id),
                team_a=loser.name, score_a=loser.score, team_b=winner.name, score_b=winner.score,
                big_stat=f"{loser.score:.1f} & still lost",
                headline=f"{loser.name} dropped {loser.score:.1f} and STILL lost",
                blurb=(f"{loser.name} scored {loser.score:.1f} — above the league median of "
                       f"{league_median:.1f} — and still lost to {winner.name} ({winner.score:.1f}). "
                       f"Brutal scheduling luck."),
            )
            if unlucky is None or spice > unlucky[0]:
                unlucky = (spice, m)
        # Luckiest: won with a score below the league median.
        gap_winner = league_median - winner.score
        if gap_winner > 0:
            spice = _clamp(36 + min(34, gap_winner * 1.5))
            m = Moment(
                type=MomentType.lucky, season=season, week=week, spice=spice,
                team_id=winner.team_id, dedup_key=str(winner.team_id),
                team_a=winner.name, score_a=winner.score, team_b=loser.name, score_b=loser.score,
                big_stat=f"won with {winner.score:.1f}",
                headline=f"{winner.name} backed into a W with {winner.score:.1f}",
                blurb=(f"{winner.name} won with just {winner.score:.1f}, below the league median of "
                       f"{league_median:.1f}, but drew {loser.name} ({loser.score:.1f}). "
                       f"Schedule did the heavy lifting."),
            )
            if lucky is None or spice > lucky[0]:
                lucky = (spice, m)
    return [t[1] for t in (unlucky, lucky) if t is not None]


# ── weekly superlatives: high / low score ───────────────────────────────────
def _superlatives(sides_all, season, week) -> list[Moment]:
    if not sides_all:
        return []
    top = max(sides_all, key=lambda s: s.score)
    low = min(sides_all, key=lambda s: s.score)
    out = [
        Moment(
            type=MomentType.high_score, season=season, week=week, spice=48.0,
            team_id=top.team_id, dedup_key=str(top.team_id),
            player=None, big_stat=f"{top.score:.1f}",
            team_a=top.name, score_a=top.score,
            headline=f"{top.name} hung the week's high: {top.score:.1f}",
            blurb=f"{top.name} posted the top score of Week {week}: {top.score:.1f} points.",
        ),
        Moment(
            type=MomentType.low_score, season=season, week=week, spice=42.0,
            team_id=low.team_id, dedup_key=str(low.team_id),
            player=None, big_stat=f"{low.score:.1f}",
            team_a=low.name, score_a=low.score,
            headline=f"{low.name} stunk up the week: {low.score:.1f}",
            blurb=f"{low.name} put up the worst score of Week {week}: a measly {low.score:.1f} points.",
        ),
    ]
    return out


# ── biggest points left on the bench ────────────────────────────────────────
def _bench_blunder(sides_all, season, week) -> list[Moment]:
    best = None  # (delta, side, started_player, bench_player)
    for side in sides_all:
        bench = _bench(side)
        if not bench:
            continue
        for s in _starters(side):
            s_pts = _f(getattr(s, "points", 0.0))
            slot = getattr(s, "slot_position", None)
            # Bench players eligible for this starter's slot who outscored them.
            cands = [b for b in bench
                     if slot in (getattr(b, "eligibleSlots", []) or [])
                     and _f(getattr(b, "points", 0.0)) > s_pts]
            if not cands:
                continue
            b = max(cands, key=lambda p: _f(getattr(p, "points", 0.0)))
            delta = _f(getattr(b, "points", 0.0)) - s_pts
            if best is None or delta > best[0]:
                best = (delta, side, s, b)
    if best is None or best[0] < settings.content_bench_blunder_min:
        return []
    delta, side, started, benched = best
    bn, sn = getattr(benched, "name", "?"), getattr(started, "name", "?")
    bp, sp = _f(getattr(benched, "points", 0)), _f(getattr(started, "points", 0))
    return [Moment(
        type=MomentType.bench_blunder, season=season, week=week,
        spice=_clamp(40 + min(50, delta * 2)),
        team_id=side.team_id, dedup_key=f"{side.team_id}:{getattr(benched,'playerId','')}",
        player=bn, player_team=side.name, big_stat=f"-{delta:.1f}",
        headline=f"{side.name} benched {bn} ({bp:.1f}) for {sn} ({sp:.1f})",
        blurb=(f"{side.name} left {delta:.1f} points on the bench in Week {week} — "
               f"{bn} dropped {bp:.1f} while {sn} ({sp:.1f}) started in their spot."),
    )]


# ── boom / bust vs projection ────────────────────────────────────────────────
def _boom_bust(sides_all, season, week) -> list[Moment]:
    boom = None  # (delta, side, player)
    bust = None  # (deficit, side, player)
    for side in sides_all:
        for p in _starters(side):
            pts = _f(getattr(p, "points", 0.0))
            proj = _f(getattr(p, "projected_points", 0.0))
            over = pts - proj
            if proj > 0 and (boom is None or over > boom[0]):
                boom = (over, side, p)
            # Only count busts for players actually expected to produce.
            if proj >= 8 and (bust is None or (proj - pts) > bust[0]):
                bust = (proj - pts, side, p)
    out: list[Moment] = []
    if boom is not None and boom[0] >= 6:
        over, side, p = boom
        pts, proj = _f(getattr(p, "points", 0)), _f(getattr(p, "projected_points", 0))
        nm = getattr(p, "name", "?")
        out.append(Moment(
            type=MomentType.boom, season=season, week=week, spice=_clamp(30 + min(45, over * 1.6)),
            team_id=side.team_id, dedup_key=f"boom:{getattr(p,'playerId','')}",
            player=nm, player_team=side.name, big_stat=f"{pts:.1f}",
            headline=f"{nm} went nuclear: {pts:.1f} ({over:+.1f} vs proj)",
            blurb=(f"{nm} ({side.name}) smashed projections in Week {week} — {pts:.1f} points "
                   f"against a {proj:.1f} projection ({over:+.1f})."),
        ))
    if bust is not None and bust[0] >= 6:
        deficit, side, p = bust
        pts, proj = _f(getattr(p, "points", 0)), _f(getattr(p, "projected_points", 0))
        nm = getattr(p, "name", "?")
        out.append(Moment(
            type=MomentType.bust, season=season, week=week, spice=_clamp(30 + min(42, deficit * 1.6)),
            team_id=side.team_id, dedup_key=f"bust:{getattr(p,'playerId','')}",
            player=nm, player_team=side.name, big_stat=f"{pts:.1f}",
            headline=f"{nm} laid an egg: {pts:.1f} ({-deficit:+.1f} vs proj)",
            blurb=(f"{nm} ({side.name}) busted hard in Week {week} — projected {proj:.1f}, "
                   f"managed only {pts:.1f} ({-deficit:+.1f})."),
        ))
    return out
