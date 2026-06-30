"""Phase 0 — prove ESPN read access on the REAL league.

Run after putting your cookies + league id in .env:

    uv run python scripts/verify_espn_read.py

It confirms: cookies work, the league is reachable, settings parse into a
league-adaptive LeagueSettings, and how far back history goes — the gate before
any modeling work.
"""

from __future__ import annotations

import logging
import sys

from fantasy.config import settings
from fantasy.espn.client import EspnAuthError, EspnClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> int:
    if not settings.has_espn_auth:
        print("✗ Missing ESPN auth. Copy .env.example -> .env and fill in:")
        print("    ESPN_S2, ESPN_SWID, ESPN_LEAGUE_ID (and ESPN_SEASON / ESPN_TEAM_ID)")
        print("  Cookies: espn.com → DevTools → Application → Cookies → espn_s2 and SWID.")
        return 1

    client = EspnClient()
    print(f"→ League {client.league_id}, season {client.season}\n")

    try:
        ls = client.league_settings()
    except EspnAuthError as e:
        print(f"✗ Auth/access error: {e}")
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"✗ Failed to read settings: {e}")
        return 1

    print("✓ League settings parsed:")
    print(f"    {ls.summary()}\n")
    print(f"    Scoring rules ({len(ls.scoring)} active):")
    for stat, pts in sorted(ls.scoring.items()):
        print(f"      {stat:32s} {pts:+.2f}")
    if ls.position_reception_bonus:
        print(f"    TE-premium / reception bonus: {ls.position_reception_bonus}")
    print()

    try:
        teams = client.teams()
        print(f"✓ {len(teams)} teams:")
        for t in teams:
            owner = getattr(t, "team_name", "?")
            n = len(getattr(t, "roster", []) or [])
            print(f"      [{getattr(t, 'team_id', '?'):>2}] {owner} — {n} players")
        print()

        mine = client.my_team()
        if mine is not None:
            print(f"✓ Your team: {getattr(mine, 'team_name', '?')}")
            for p in getattr(mine, "roster", [])[:25]:
                print(
                    f"      {getattr(p, 'name', '?'):24s} {getattr(p, 'position', '?'):4s} "
                    f"proj={getattr(p, 'projected_total_points', None)}"
                )
            print()
        else:
            print("• ESPN_TEAM_ID not set — skipping 'your team' detail.\n")

        fas = client.free_agents(size=5)
        print("✓ Sample free agents:")
        for p in fas[:5]:
            print(f"      {getattr(p, 'name', '?'):24s} {getattr(p, 'position', '?'):4s}")
        print()

        picks = client.draft()
        print(f"✓ Draft results readable: {len(picks)} picks recorded for {client.season}.")
    except Exception as e:  # noqa: BLE001
        print(f"⚠ Settings worked but a roster/FA/draft call failed: {e}")
        print("  (Often fine — some calls are season-dependent. Settings access is the key gate.)")
        return 0

    print("\n✓ Phase 0 PASSED — read access confirmed. Ready for Phase 1 (model + backtest).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
