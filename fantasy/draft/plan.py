"""Draft Plan — a positional draft strategy computed from THIS league's rules.

The plan is *strategy*, not player picks: what each round is for, which positions
carry the most edge under the league's exact scoring, and which slots to stream
rather than draft. Every number is derived from the merged
:class:`~fantasy.league_settings.LeagueSettings` and the season value board
(:func:`fantasy.draft.season_board.build_season_board`) — nothing is hardcoded to
the user's league, so the same function produces a sensible plan for standard PPR,
superflex, IDP, HC, or any mix of the customizable ESPN rules.

Strategy model
--------------
VBD (value over replacement) ranks players across positions; the round plan walks
a 12-team snake and, at each of *my* expected picks, asks which positions still
have real value on the board (survival-weighted by ADP) and where the tier cliffs
fall. Research-derived *gates* provide the phrasing and the guardrails on top of
that math — they come from the completed research pass documented in the plan
memo, not from live tuning:

- **Return yards are the format's biggest edge.** Consensus projections carry ZERO
  return yardage, so a full-time returner (≈350-500 pts/season at 0.25/yd) is
  invisible to every ADP-based opponent — the plan surfaces them explicitly and
  highlights returner-boosted players in the mid-round window.
- **QB gate:** never before round 6 in a 1-QB league (replacement QB is cheap);
  the gate lifts entirely in superflex.
- **K / DP gate:** last two rounds only (flat replacement — an elite MIKE LB beats
  a waiver LB by only ~4-6 pts/wk, so there's no reason to reach).
- **HC gate:** the literal last pick, or skip-and-stream (HC is pure weekly EV —
  streaming the biggest moneyline favorite beats holding the best drafted coach).
- **TE:** only worth a mid-round pick if an elite-tier TE is actually available;
  otherwise stream/wait.
- **Slight RB tilt in rounds 1-2** (the RB scoring curve is the steepest, and 2
  FLEX deepens the RB/WR startable pool).

Determinism: given a fixed board, ``build_draft_plan`` is a pure function — all
sorting is total and all numbers are rounded to 0.1, so tests can assert on exact
output.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from fantasy.espn.stat_ids import IDP_POSITIONS
from fantasy.league_settings import LeagueSettings
from fantasy.valuation.hc import hc_stream_ev
from fantasy.valuation.vor import (
    pooled_position,
    replacement_baselines,
    replacement_counts,
)

# ── strategy constants (from the research pass; see module docstring) ──────────
# A standard-PPR reference the rules_impact deltas are measured against.
_STD_PPR_TARGET_PTS = 0.0  # standard PPR doesn't score targets
# Research anchor for the per-target headline: an alpha WR sees ~170 targets/yr.
_ALPHA_WR_TARGETS = 170.0
# Positions that get an explicit per-position value row (offense + special slots).
_CORE_POSITIONS = ["QB", "RB", "WR", "TE", "K"]
# The mid-round window where returner-hybrids are the sharpest edge.
_RETURNER_WINDOW = (6, 8)
# A material rules-impact deviation must move at least this many season pts to list.
_MATERIAL_PTS = 5.0
# Elite-TE gate: a TE is "mid-round worthy" if its VOR clears this vs the field.
_ELITE_TE_VOR = 25.0
# Round after which a 1-QB league may take its starting QB.
_QB_EARLIEST_ROUND = 6


def _round1(x) -> float:
    """Round to 0.1, coercing NaN/None to 0.0 (keeps output JSON-clean + stable)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(v) or math.isinf(v):
        return 0.0
    return round(v, 1)


def _starters_string(league: LeagueSettings) -> str:
    """'1QB / 2RB / 2WR / 1TE / 2FLEX / 1K / 1DP / 1HC' from the roster slots."""
    order = ["QB", "TQB", "RB", "RB/WR", "WR", "WR/TE", "TE", "FLEX", "OP",
             "DL", "LB", "DB", "DP", "D/ST", "K", "P", "HC", "ER"]
    slots = league.roster.starter_slots
    parts = []
    for slot in order:
        n = slots.get(slot, 0)
        if n:
            parts.append(f"{n}{slot}")
    # Any slot not in the canonical order (unlikely) tacked on the end.
    for slot, n in slots.items():
        if slot not in order and n:
            parts.append(f"{n}{slot}")
    return " / ".join(parts)


def _survival_available(board: pd.DataFrame, pick: int, horizon: int | None = None
                        ) -> pd.DataFrame:
    """Players expected to still be on the board at overall ``pick``.

    A player is 'available' if their ADP hasn't clearly passed the pick — using a
    half-sigma cushion (``adp >= pick - 0.5*adp_sd``) so a player whose ADP is a
    hair before the pick, but with wide variance, still counts as reachable.

    ``horizon`` optionally bounds the pool ABOVE (``adp <= pick + horizon``) so a
    round's "best available" reflects players realistically in reach *around* this
    pick rather than anyone who happens to still be on the board — this is what
    lets the round-by-round VOR actually decrease as the draft progresses. It is
    only applied to players the market prices (finite ADP); consensus-invisible
    return-value players keep a default late ADP and are surfaced separately.
    """
    adp = pd.to_numeric(board["adp"], errors="coerce")
    sd = pd.to_numeric(board["adp_sd"], errors="coerce").fillna(0.0)
    mask = adp >= (pick - 0.5 * sd)
    if horizon is not None:
        mask = mask & (adp <= (pick + horizon))
    return board[mask]


def _pick_overall(round_no: int, slot: int, team_count: int) -> int:
    """My expected overall pick in a snake draft at ``round_no`` from ``slot``."""
    if round_no % 2 == 1:  # odd rounds go 1..T
        return (round_no - 1) * team_count + slot
    return round_no * team_count - slot + 1  # even rounds snake back T..1


# ── league summary ─────────────────────────────────────────────────────────────
def _league_summary(league: LeagueSettings) -> str:
    starters = _starters_string(league)
    bits = []
    tgt = float(league.scoring.get("receiving_targets", 0.0) or 0.0)
    if tgt:
        bits.append(f"{tgt:g} pts per reception target")
    kr = float(league.scoring.get("kickoff_return_yards", 0.0) or 0.0)
    pr = float(league.scoring.get("punt_return_yards", 0.0) or 0.0)
    if kr or pr:
        yd = kr or pr
        bits.append(f"{yd:g} pts per return yard (kick + punt)")
    slots = league.roster.starter_slots
    flex_slots = sum(n for s, n in slots.items() if s in ("FLEX", "RB/WR", "WR/TE"))
    if flex_slots >= 2:
        bits.append(f"{flex_slots} FLEX")
    if any(slots.get(s, 0) for s in ("DP",)) or any(
        slots.get(s, 0) for s in IDP_POSITIONS
    ):
        bits.append("an individual defensive player (DP) slot instead of D/ST")
    if slots.get("HC", 0):
        win = float(league.scoring.get("hc_team_win", 0.0) or 0.0)
        loss = float(league.scoring.get("hc_team_loss", 0.0) or 0.0)
        bits.append(f"a head-coach slot ({win:+g} win / {loss:+g} loss)")
    rec = float(league.scoring.get("receptions", 0.0) or 0.0)
    base = f"{league.scoring_format.value.replace('_', '-')}"
    if rec and abs(rec - 0.5) < 1e-9:
        base = "half-PPR"
    custom = (" It layers on " + "; ".join(bits) + ".") if bits else ""
    return (
        f"{league.team_count}-team league starting {starters} on a {base} base."
        f"{custom} These rules shift value in ways consensus rankings don't price, "
        f"which is exactly where this plan finds its edge."
    )


# ── rules impact ───────────────────────────────────────────────────────────────
def _impact_targets(league: LeagueSettings, board: pd.DataFrame) -> dict | None:
    """Per-target scoring: alpha-WR season delta from the board's real targets."""
    pts = float(league.scoring.get("receiving_targets", 0.0) or 0.0)
    if abs(pts - _STD_PPR_TARGET_PTS) < 1e-9:
        return None
    wr = board[board["position"] == "WR"]
    tgt_col = "receiving_targets"
    if tgt_col in board.columns and wr[tgt_col].notna().any():
        top_targets = float(pd.to_numeric(wr[tgt_col], errors="coerce").dropna()
                            .sort_values(ascending=False).head(3).mean())
    else:
        top_targets = _ALPHA_WR_TARGETS
    season_delta = top_targets * pts
    ppg = season_delta / 17.0
    return {
        "rule": "Points per target",
        "headline": f"{pts:g}/target adds ~{_round1(ppg)} ppg to alpha WRs",
        "detail": (
            f"An alpha WR on ~{int(round(top_targets))} targets banks "
            f"~{_round1(season_delta)} extra season pts here — high-volume, "
            f"low-efficiency receivers (and pass-catching backs) gain the most; "
            f"target-starved deep threats gain the least."
        ),
        "magnitude_pts": _round1(season_delta),
    }


def _impact_returns(league: LeagueSettings, board: pd.DataFrame) -> dict | None:
    """Return yards: top-3 returner values from the board's return_pts column."""
    kr = float(league.scoring.get("kickoff_return_yards", 0.0) or 0.0)
    pr = float(league.scoring.get("punt_return_yards", 0.0) or 0.0)
    if not kr and not pr:
        return None
    rp = pd.to_numeric(board.get("return_pts"), errors="coerce").fillna(0.0)
    top3 = rp.sort_values(ascending=False).head(3)
    top3 = top3[top3 > 0]
    if top3.empty:
        best = 0.0
        avg = 0.0
    else:
        best = float(top3.max())
        avg = float(top3.mean())
    return {
        "rule": "Return yardage",
        "headline": f"Full-time returners are worth ~{_round1(avg)} hidden season pts",
        "detail": (
            f"Your league scores return yards, but every consensus ranking carries "
            f"ZERO of it — the top returner projects ~{_round1(best)} season pts "
            f"({_round1(best / 17.0)} ppg) that nobody else at the table is pricing. "
            f"This is the single biggest edge in the format; target confirmed "
            f"full-time returners who also hold an offensive role."
        ),
        "magnitude_pts": _round1(avg),
    }


def _impact_two_flex(league: LeagueSettings) -> dict | None:
    """2+ FLEX: replacement-rank shift vs a 1-FLEX version of the same roster."""
    slots = league.roster.starter_slots
    flex_n = sum(n for s, n in slots.items() if s in ("FLEX", "RB/WR", "WR/TE"))
    if flex_n < 2:
        return None
    counts_now = replacement_counts(league)
    # Same roster with one fewer flex — drop from whichever flex-style slot the
    # league actually uses (a 2-flex league may run RB/WR + WR/TE with no FLEX).
    one_flex = league.model_copy(deep=True)
    new_slots = dict(one_flex.roster.slots)
    for flex_slot in ("FLEX", "RB/WR", "WR/TE"):
        if new_slots.get(flex_slot, 0) >= 1:
            new_slots[flex_slot] = new_slots[flex_slot] - 1
            break
    one_flex.roster.slots = new_slots
    counts_one = replacement_counts(one_flex)
    rb_shift = counts_now.get("RB", 0) - counts_one.get("RB", 0)
    wr_shift = counts_now.get("WR", 0) - counts_one.get("WR", 0)
    return {
        "rule": f"{flex_n} FLEX",
        "headline": f"{flex_n} FLEX deepens the startable RB/WR pool",
        "detail": (
            f"Replacement level drops ~{rb_shift} RB and ~{wr_shift} WR ranks vs a "
            f"single-FLEX roster ({counts_now.get('RB', 0)} startable RB, "
            f"{counts_now.get('WR', 0)} WR league-wide) — depth wins, and the RB "
            f"scoring curve is steepest, so lean RB slightly earlier."
        ),
        "magnitude_pts": _round1(max(rb_shift, wr_shift)),
    }


def _impact_dp(league: LeagueSettings, board: pd.DataFrame) -> dict | None:
    """DP: LB-dominance + flat replacement (elite vs waiver ≈ 4-6 pts/wk)."""
    slots = league.roster.starter_slots
    has_dp = slots.get("DP", 0) or any(slots.get(s, 0) for s in IDP_POSITIONS)
    if not has_dp:
        return None
    dp = board[board["position"].map(pooled_position) == "DP"]
    proj = pd.to_numeric(dp.get("proj"), errors="coerce").dropna().sort_values(ascending=False)
    if len(proj) >= 15:
        elite = float(proj.head(3).mean())
        waiver = float(proj.iloc[12:15].mean())
        weekly = (elite - waiver) / 17.0
    else:
        weekly = 5.0  # research anchor when the board is thin
    return {
        "rule": "Individual defensive player (DP)",
        "headline": f"Elite DP beats a waiver DP by only ~{_round1(weekly)} pts/wk",
        "detail": (
            "Replacement is flat and LB-dominated (every-down MIKE linebackers rack "
            "up tackles). Don't reach — draft the best every-down LB in the last two "
            "rounds and hold. The points-per-pick you'd spend reaching are worth more "
            "on RB/WR depth."
        ),
        "magnitude_pts": _round1(weekly),
    }


def _impact_hc(league: LeagueSettings) -> dict | None:
    """HC: stream EV from hc_stream_ev."""
    if not league.roster.starter_slots.get("HC", 0):
        return None
    ev = hc_stream_ev(league)
    edge = ev["stream_season_pts"] - ev["best_drafted_season_pts"]
    return {
        "rule": "Head coach (HC)",
        "headline": f"Streaming a coach beats drafting one by ~{_round1(edge)} season pts",
        "detail": (
            f"HC is pure weekly EV. Streaming the biggest moneyline favorite each week "
            f"projects ~{_round1(ev['stream_weekly_pts'])} pts/wk "
            f"(~{_round1(ev['stream_season_pts'])} season) vs "
            f"~{_round1(ev['best_drafted_weekly_pts'])}/wk "
            f"(~{_round1(ev['best_drafted_season_pts'])}) for holding the best drafted "
            f"coach. Spend your last pick on the Week-1 biggest favorite, or skip and "
            f"claim off waivers."
        ),
        "magnitude_pts": _round1(edge),
    }


def _rules_impact(league: LeagueSettings, board: pd.DataFrame) -> list[dict]:
    """Every MATERIAL deviation from standard PPR, phrased with research heuristics.

    Only rules active in THIS league appear; each is skipped if inactive or below
    the materiality floor. Order is fixed (targets, returns, flex, DP, HC) so the
    output is deterministic.
    """
    out = []
    for entry in (
        _impact_targets(league, board),
        _impact_returns(league, board),
        _impact_two_flex(league),
        _impact_dp(league, board),
        _impact_hc(league),
    ):
        if entry is None:
            continue
        # An active special-rule entry (returns/flex/DP/HC) is always shown — its
        # presence IS the strategy signal, and its magnitude may be small by nature
        # (e.g. DP's weekly edge). Only the per-target headline is magnitude-gated,
        # since a trivial per-target value genuinely doesn't move draft strategy.
        always = (
            entry["rule"] in ("Return yardage", "Individual defensive player (DP)",
                              "Head coach (HC)")
            or entry["rule"].endswith("FLEX")
        )
        if always or abs(entry["magnitude_pts"]) >= _MATERIAL_PTS:
            out.append(entry)
    return out


# ── positional value ───────────────────────────────────────────────────────────
def _positional_value(league: LeagueSettings, board: pd.DataFrame) -> list[dict]:
    """Per-position replacement rank/points, top-3 avg, and tier dropoff."""
    counts = replacement_counts(league)
    baselines = replacement_baselines(board, league)
    pooled = board["position"].map(pooled_position)
    # Positions to report: core offense + K, plus any special pool the league uses.
    positions = list(_CORE_POSITIONS)
    slots = league.roster.starter_slots
    if slots.get("DP", 0) or any(slots.get(s, 0) for s in IDP_POSITIONS):
        positions.append("DP")
    if slots.get("HC", 0):
        positions.append("HC")

    rows = []
    for pos in positions:
        rank = counts.get(pos, 0)
        repl = baselines.get(pos, 0.0)
        vals = (pd.to_numeric(board.loc[pooled == pos, "proj"], errors="coerce")
                .dropna().sort_values(ascending=False).to_numpy())
        if len(vals) == 0:
            continue
        top3 = float(np.mean(vals[:3])) if len(vals) >= 1 else 0.0
        dropoff = top3 - repl
        # Tier note: steep dropoff -> scarce; flat -> streamable.
        if pos in ("DP", "HC", "K"):
            note = "flat — stream/late-round, don't reach"
        elif dropoff >= 60:
            note = "steep tier cliff — value concentrates at the top"
        elif dropoff >= 30:
            note = "moderate scarcity — secure a starter mid-draft"
        else:
            note = "deep — startable value lasts into later rounds"
        rows.append({
            "position": pos,
            "starters_league_wide": int(rank),
            "replacement_rank": int(rank),
            "replacement_pts": _round1(repl),
            "top3_avg_pts": _round1(top3),
            "dropoff_pts": _round1(dropoff),
            "tier_note": note,
        })
    return rows


# ── round plan ─────────────────────────────────────────────────────────────────
def _best_vor_by_position(avail: pd.DataFrame) -> dict[str, dict]:
    """For each pooled position among ``avail``: best VOR + next-tier dropoff."""
    if avail.empty:
        return {}
    a = avail.copy()
    a["pool"] = a["position"].map(pooled_position)
    a["vor_n"] = pd.to_numeric(a["vor"], errors="coerce")
    out: dict[str, dict] = {}
    for pool, grp in a.groupby("pool"):
        vals = grp["vor_n"].dropna().sort_values(ascending=False).to_numpy()
        if len(vals) == 0:
            continue
        best = float(vals[0])
        # Dropoff to the tier ~8 picks deeper (a run's worth) at this position.
        deeper = float(vals[min(len(vals) - 1, 4)])
        out[pool] = {"best_vor": best, "dropoff": max(best - deeper, 0.0),
                     "count": int(len(vals))}
    return out


def _round_plan(league: LeagueSettings, board: pd.DataFrame, my_slot: int) -> list[dict]:
    """One entry per round with survival-aware positional priorities + gates."""
    T = league.team_count
    total = league.roster.total_starters + league.roster.bench_size
    total = max(total, 1)
    superflex = league.roster.has_superflex
    slots = league.roster.starter_slots
    has_dp = bool(slots.get("DP", 0) or any(slots.get(s, 0) for s in IDP_POSITIONS))
    has_hc = bool(slots.get("HC", 0))
    has_k = bool(slots.get("K", 0))
    scores_returns = bool(
        float(league.scoring.get("kickoff_return_yards", 0.0) or 0.0)
        or float(league.scoring.get("punt_return_yards", 0.0) or 0.0)
    )
    ret_map = (pd.to_numeric(board.get("return_pts"), errors="coerce").fillna(0.0)
               if "return_pts" in board.columns else None)

    plan = []
    for r in range(1, total + 1):
        pick = _pick_overall(r, my_slot, T)
        rounds_left = total - r + 1
        last_two = rounds_left <= 2
        is_last = r == total
        # "Around this pick": from here through ~1.5 rounds of picks forward, so the
        # best-available VOR reflects a draft-realistic tier and erodes each round.
        priorities: list[dict] = []
        avoid: list[str] = []
        note = ""

        # ── QB gate ──
        qb_ok = superflex or r >= _QB_EARLIEST_ROUND
        if not qb_ok:
            avoid.append("QB")

        # ── K / DP gate: last two rounds only ──
        if not last_two:
            if has_k:
                avoid.append("K")
            if has_dp:
                avoid.append("DP")

        def _skill_priorities(best: dict) -> list[dict]:
            """Ranked, gated skill-position priorities from a best-VOR dict."""
            skill = {p: best[p] for p in ("QB", "RB", "WR", "TE") if p in best}

            # RB tilt in rounds 1-2 (steepest curve, deepened by extra FLEX).
            def _key(item):
                pos, info = item
                bump = 1.05 if (pos == "RB" and r <= 2) else 1.0
                return info["best_vor"] * bump

            out: list[dict] = []
            for pos, info in sorted(skill.items(), key=_key, reverse=True):
                if pos == "QB" and not qb_ok:
                    continue
                # Below replacement = waiver fodder, not a priority.
                if info["best_vor"] <= 0:
                    continue
                if pos == "TE" and info["best_vor"] < _ELITE_TE_VOR:
                    # TE is only worth a pick when an elite-tier one is on the
                    # board; otherwise stream/wait rather than spend a pick.
                    continue
                reason = (f"best {pos} VOR ≈ {_round1(info['best_vor'])}"
                          + (f", tier cliff of {_round1(info['dropoff'])} pts within "
                             f"~8 picks" if info["dropoff"] >= 8 else ""))
                out.append({"position": pos, "reason": reason})
                if len(out) >= 3:
                    break
            return out

        # "Around this pick": from here through ~1.5 rounds of picks forward, so
        # the best-available VOR reflects a draft-realistic tier and erodes each
        # round. Past the market's ADP coverage that window holds nothing worth
        # starting — fall back to the full remaining pool (late rounds are
        # best-VOR-available by definition).
        avail = _survival_available(board, pick, horizon=int(round(1.5 * T)))
        priorities = _skill_priorities(_best_vor_by_position(avail))
        if not priorities:
            priorities = _skill_priorities(
                _best_vor_by_position(_survival_available(board, pick)))

        # ── returner-boosted highlight in the mid-round window ──
        # Search the FULL available pool (not the horizon-bounded one): the whole
        # point is that consensus-invisible returners carry a late/default ADP, so
        # they're reachable here precisely because the market isn't drafting them.
        if scores_returns and ret_map is not None and _RETURNER_WINDOW[0] <= r <= _RETURNER_WINDOW[1]:
            full_avail = _survival_available(board, pick)
            ravail = full_avail.assign(_rp=ret_map.reindex(full_avail.index).fillna(0.0))
            ravail = ravail[ravail["_rp"] > 0]
            if not ravail.empty:
                top = ravail.sort_values("_rp", ascending=False).iloc[0]
                note = (f"Returner window — {top['name']} carries "
                        f"~{_round1(top['_rp'])} hidden return pts nobody else prices.")

        # ── last-two-rounds special slots ──
        # When HC is drafted, reserve the literal last pick for it and put DP + K in
        # the second-to-last round; otherwise DP + K fill the final two picks. The
        # specials are the actual recommendation in these rounds, so they LEAD the
        # list (skill depth is the alternate) — the final [:3] cap keeps them.
        if last_two:
            dp_rec = {"position": "DP",
                      "reason": "draft the best every-down MIKE LB and hold"}
            k_rec = {"position": "K",
                     "reason": "kicker is fungible — take the best one and move on"}
            others = ([dp_rec] if has_dp else []) + ([k_rec] if has_k else [])
            specials: list[dict] = []
            if has_hc:
                # HC takes the literal last pick; DP + K share the round before.
                if rounds_left == 2:
                    specials = others
            elif len(others) == 2:
                # No HC: DP second-to-last, K with the literal last pick.
                specials = [others[0]] if rounds_left == 2 else [others[1]]
            elif others and rounds_left == 1:
                # Only one special slot: it gets the last pick.
                specials = others
            priorities = specials + priorities
        # ── HC: literal last pick or skip-and-stream ──
        if has_hc:
            if is_last:
                priorities = ([{"position": "HC",
                                "reason": "spend the last pick on the Week-1 biggest favorite"}]
                              + [p for p in priorities if p["position"] != "HC"])
            elif rounds_left <= 3 and not any(p["position"] == "HC" for p in priorities):
                note = note or "Skip HC here and stream the biggest weekly favorite off waivers."

        # ── fallback note ──
        if not priorities and not note:
            note = "Best player available — no position is clearly ahead at this pick."
        if r <= 2 and any(p["position"] == "RB" for p in priorities):
            note = note or "Slight RB tilt: the RB scoring curve is steepest early."

        plan.append({
            "round": r,
            "pick_overall": int(pick),
            "priorities": priorities[:3],
            "avoid": avoid,
            "note": note,
        })
    return plan


# ── streamability ──────────────────────────────────────────────────────────────
def _streamability(league: LeagueSettings, board: pd.DataFrame) -> list[dict]:
    """Per started-slot streaming verdict with an EV note."""
    slots = league.roster.starter_slots
    superflex = league.roster.has_superflex
    out = []
    seen = set()

    def _add(slot, verdict, note):
        if slot in seen:
            return
        seen.add(slot)
        out.append({"slot": slot, "verdict": verdict, "ev_note": note})

    for slot, n in slots.items():
        if not n:
            continue
        if slot == "HC":
            ev = hc_stream_ev(league)
            _add("HC", "stream_weekly",
                 f"stream the biggest weekly favorite (~{_round1(ev['stream_season_pts'])} "
                 f"season pts vs ~{_round1(ev['best_drafted_season_pts'])} drafted)")
        elif slot == "DP" or slot in IDP_POSITIONS:
            _add("DP", "draft_hold",
                 "draft one every-down MIKE LB in the last two rounds and hold "
                 "(elite vs waiver ≈ 4-6 pts/wk — not worth chasing weekly)")
        elif slot == "K":
            _add("K", "late_hold", "draft in the last two rounds; only stream on a bye")
        elif slot in ("QB", "TQB"):
            verdict = "draft_hold" if superflex else "set_and_forget"
            _add("QB", verdict,
                 "superflex — QB is scarce, draft two" if superflex
                 else "1-QB: draft one starter mid-draft and forget it")
        elif slot == "TE":
            _add("TE", "set_and_forget",
                 "grab a starter (elite tier if it falls to you) and hold")
        elif slot in ("RB", "WR", "FLEX", "RB/WR", "WR/TE", "OP"):
            _add(slot, "draft_hold",
                 "depth wins — the extra weekly skill starters drain the wire, so "
                 "hoard RB/WR")
    return out


# ── returner watch ─────────────────────────────────────────────────────────────
def _returner_watch(league: LeagueSettings, board: pd.DataFrame) -> dict | None:
    """Top return-value players joined with current KR/PR role, if returns scored."""
    kr = float(league.scoring.get("kickoff_return_yards", 0.0) or 0.0)
    pr = float(league.scoring.get("punt_return_yards", 0.0) or 0.0)
    if not kr and not pr:
        return None
    if "return_pts" not in board.columns:
        return {"note": _RETURNER_NOTE, "players": []}

    rp = board.assign(rp_pts=pd.to_numeric(board["return_pts"], errors="coerce").fillna(0.0))
    rp = rp[rp["rp_pts"] > 0].sort_values("rp_pts", ascending=False).head(10)

    roles = _returner_roles(league.season)
    players = []
    for r in rp.itertuples(index=False):
        pid = getattr(r, "player_id", None)
        role = roles.get(pid, "returner")
        players.append({
            "name": getattr(r, "name", None) or str(pid),
            "team": getattr(r, "team", None),
            "position": getattr(r, "position", None),
            "role": role,
            "return_pts": _round1(getattr(r, "rp_pts")),
        })
    return {"note": _RETURNER_NOTE, "players": players}


_RETURNER_NOTE = (
    "Return yardage is the format's biggest edge — nobody else's rankings price it. "
    "Prioritize confirmed full-time returners who also hold an offensive role, but "
    "mind job security: return duty flips week to week. Round out the bench ~60/40 "
    "RB/WR, and value handcuffs who double as kick returners."
)


def _returner_roles(season: int | None) -> dict[str, str]:
    """gsis_id -> 'KR'/'PR'/'KR+PR' from the current depth-chart holders (best effort)."""
    if season is None:
        return {}
    try:
        from fantasy.data.returns import current_returners

        holders = current_returners(season)
    except Exception:  # noqa: BLE001 — role guidance is a nicety, never blocks the plan
        return {}
    roles: dict[str, str] = {}
    if holders is None or holders.empty:
        return roles
    for r in holders.itertuples(index=False):
        pid = getattr(r, "gsis_id", None)
        role = getattr(r, "role", None)
        if pid is None or role is None:
            continue
        if pid in roles and roles[pid] != role:
            roles[pid] = "KR+PR"
        else:
            roles[pid] = role
    return roles


# ── public entrypoint ──────────────────────────────────────────────────────────
def build_draft_plan(
    league: LeagueSettings,
    season: int,
    board: pd.DataFrame | None = None,
    my_slot: int | None = None,
) -> dict:
    """Build the full draft plan for ``league`` in ``season``.

    ``board`` is a season value board (:func:`build_season_board`); if omitted it
    is built here. ``my_slot`` is my snake-draft position (1..team_count); if
    omitted, a mid-slot (``team_count // 2``) is assumed.

    Returns a JSON-serializable dict with the sections documented in the module
    docstring. Deterministic given a fixed ``board``.
    """
    if board is None:
        from fantasy.draft.season_board import build_season_board

        board = build_season_board(season, league)

    board = board.copy()
    # Defensive column presence so the plan builds against any board shape.
    for col in ("position", "proj", "vor", "adp", "adp_sd", "return_pts",
                "player_id", "name", "team"):
        if col not in board.columns:
            board[col] = np.nan if col not in ("position", "name", "team") else None
    board = board[board["position"].notna()].reset_index(drop=True)

    slot = my_slot if my_slot else max(1, league.team_count // 2)

    return {
        "league_summary": _league_summary(league),
        "rules_impact": _rules_impact(league, board),
        "positional_value": _positional_value(league, board),
        "round_plan": _round_plan(league, board, slot),
        "streamability": _streamability(league, board),
        "returner_watch": _returner_watch(league, board),
    }
