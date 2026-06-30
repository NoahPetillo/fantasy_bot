"""Back-in-time decision audit — your real season vs. what the model would do.

Thin CLI over fantasy.analysis.audit. Reconstructs every executed transaction by
diffing weekly ESPN box-score rosters, scores each on realized league points,
grades start/sit (points left on the bench), and overlays the model (best
available FA each week + win-win trades it would have proposed).

    uv run python scripts/decision_audit.py [--no-model] [--through 17]

Honest scope: EXECUTED moves only — declined/vetoed trade OFFERS aren't
retrievable from ESPN's read API. "Asset" values production; "started" is the
start/sit-aware view.
"""

from __future__ import annotations

import argparse
import logging

from fantasy.analysis.audit import season_report
from fantasy.config import settings as app_settings
from fantasy.espn.client import EspnClient

logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(name)s: %(message)s")

SEASON = 2025
TRAIN = [2021, 2022, 2023, 2024]


def fmt(x: float) -> str:
    return f"{x:+.1f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-model", action="store_true")
    ap.add_argument("--through", type=int, default=17)
    args = ap.parse_args()

    client = EspnClient(season=SEASON)
    league = client.league_settings()
    lg = client.league()
    my_id = app_settings.espn_team_id
    me = {t.team_id: t.team_name for t in lg.teams}.get(my_id, my_id)

    service = None
    if not args.no_model:
        print(f"Fitting projection model on {TRAIN} (point-in-time) ...")
        from fantasy.projections.service import ProjectionService
        service = ProjectionService(league).fit(TRAIN)

    print("Reconstructing weekly rosters from box scores ...")
    rep = season_report(client, league, my_id, SEASON, service=service, through=args.through)
    reg_end = rep["reg_end"]

    print(f"\n╔══ DECISION AUDIT — {getattr(lg.settings, 'name', '?')} {SEASON} ══╗")
    print(f"Team: {me} (id {my_id}) | reg season ends wk {reg_end}\n")

    # Trades made
    print("═" * 78 + "\nTRADES YOU MADE\n" + "═" * 78)
    if not rep["trades_made"]:
        print(f"  You made ZERO trades all season. (The league executed {rep['league_trades']} "
              f"real trades — you sat out the trade market entirely.)")
    for t in rep["trades"]:
        print(f"  Wk {t['week']}: got {', '.join(t['received'])} / sent {', '.join(t['sent'])}")

    # Waivers
    w = rep["waiver"]
    print("\n" + "═" * 78 + "\nWAIVER / FREE-AGENT MOVES YOU MADE\n" + "═" * 78)
    print(f"  (add minus drop over wk→{reg_end}; tagged per player; "
          f"asset = produced, started = only weeks you started him)\n")
    for m in w["moves"]:
        parts = [f"+{a['name']}({a['pos']}) {a['asset']:.0f}/{a['started']:.0f}st" for a in m["added"]]
        parts += [f"-{d['name']}({d['pos']}) {d['asset']:.0f}" for d in m["dropped"]]
        print(f"  Wk {m['week']:>2}  net {fmt(m['net'])}:  {', '.join(parts)}")

    # Start/sit
    ss = rep["startsit"]
    print("\n" + "═" * 78 + "\nSTART / SIT — POINTS LEFT ON YOUR BENCH\n" + "═" * 78)
    for wk in ss["weeks"]:
        line = f"  Wk {wk['week']:>2}: started {wk['started']:>6.1f} | optimal {wk['optimal']:>6.1f} | left {wk['left']:>5.1f}"
        if wk["biggest"] and wk["left"] >= 5:
            bg = wk["biggest"]
            line += f"   ⮡ start {bg['bench']} ({bg['bench_pts']}) over {bg['over']} ({bg['over_pts']})"
        print(line)
    print(f"\n  Hindsight-optimal left on bench (wk1–{reg_end}): {ss['total_left_on_bench']:.0f} pts "
          f"(~{ss['avg_per_week']:.1f}/wk) — note: nobody hits the hindsight ceiling; ~15-20/wk is normal.")
    if ss.get("model"):
        msg = ss["model"]
        verdict = "would have GAINED you" if msg["total_gain"] > 0 else "would have COST you"
        print(f"  Decision-relevant: following the MODEL's start/sit {verdict} "
              f"{abs(msg['total_gain']):.0f} realized pts (~{msg['avg_per_week']:+.1f}/wk).")

    # Model trades
    mt = rep.get("model_trades") or {}
    if mt.get("weeks"):
        print("\n" + "═" * 78 + "\nTRADES THE MODEL WOULD HAVE PROPOSED FOR YOU\n" + "═" * 78)
        for r in mt["weeks"]:
            tag = "✓" if r["realized"] > 0 else "✗"
            print(f"  Wk {r['week']:>2}: send {r['give']} → get {r['get']} "
                  f"(accept ~{r['accept']*100:.0f}%)  realized {fmt(r['realized'])} {tag}")
        print(f"\n  Across {mt['n']} weeks: positive {int(mt['win_rate']*mt['n'])}/{mt['n']}, "
              f"mean {fmt(mt['mean'])}, best {fmt(mt['best'])}, worst {fmt(mt['worst'])} pts.")

    # Bottom line
    print("\n" + "═" * 78 + "\nBOTTOM LINE\n" + "═" * 78)
    print(f"  Skill (RB/WR/TE/QB) waiver swaps — asset {fmt(w['skill_asset'])} "
          f"(started-only {fmt(w['skill_started'])})")
    print(f"  K/DST/K streaming swaps — asset {fmt(w['stream_asset'])} "
          f"(started-only {fmt(w['stream_started'])})")
    print(f"  Start/sit — left {ss['total_left_on_bench']:.0f} pts on your bench "
          f"(~{ss['avg_per_week']:.1f}/wk)")
    print(f"  Trades you made: {rep['trades_made']}.")
    if mt.get("best_trade"):
        bt = mt["best_trade"]
        print(f"  Best trade the model would have proposed: wk{bt['week']} {bt['give']} → "
              f"{bt['get']} ({fmt(bt['realized'])} realized, ~{bt['accept']*100:.0f}% accept).")
    print("\n  Caveats: 'asset' values production, not start/sit (see started-only).")
    print("  Declined/vetoed trade OFFERS aren't retrievable from ESPN's read API.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
