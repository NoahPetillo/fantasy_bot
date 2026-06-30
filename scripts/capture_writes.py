"""Phase 3 SPIKE — capture the real ESPN write request (needs cookies).

Opens your ESPN team page in a real browser (logged in via your cookies) and
records every write request (POST/PUT) while YOU manually set a lineup or do an
add/drop. The captured URL + headers + body tell us whether the write is a
replayable cookie-authenticated request (→ a tiny `requests` executor) or needs
CSRF/bearer/the full Playwright DOM tier.

Setup:
    uv sync --extra write
    uv run playwright install chromium
    uv run python scripts/capture_writes.py        # then act in the browser

Captures are written to data/captured_writes.json. This performs NO automated
writes — you drive the browser.
"""

from __future__ import annotations

import json
import sys
import time

from fantasy.config import settings

TEAM_URL = "https://fantasy.espn.com/football/team?leagueId={lid}&teamId={tid}&seasonId={season}"
WRITE_HOSTS = ("lm-api-writes", "fantasy.espn.com/apis", "/transactions", "/rosters")


def main() -> int:
    if not settings.has_espn_auth:
        print("✗ Need ESPN cookies in .env (ESPN_S2, ESPN_SWID, ESPN_LEAGUE_ID).")
        return 1
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("✗ Playwright not installed. Run: uv sync --extra write && uv run playwright install chromium")
        return 1

    captured = []

    def on_request(req):
        if req.method in ("POST", "PUT") and any(h in req.url for h in WRITE_HOSTS):
            entry = {"method": req.method, "url": req.url,
                     "headers": dict(req.headers), "post_data": req.post_data}
            captured.append(entry)
            print(f"\n📝 CAPTURED {req.method} {req.url}")
            if req.post_data:
                print(f"   body: {req.post_data[:400]}")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=False)
        ctx = browser.new_context()
        ctx.add_cookies([
            {"name": "espn_s2", "value": settings.espn_s2, "domain": ".espn.com", "path": "/"},
            {"name": "SWID", "value": settings.espn_swid_braced, "domain": ".espn.com", "path": "/"},
        ])
        page = ctx.new_page()
        page.on("request", on_request)
        url = TEAM_URL.format(lid=settings.espn_league_id,
                              tid=settings.espn_team_id or 1, season=settings.espn_season)
        page.goto(url)
        print(f"→ Opened {url}\n→ Now MANUALLY set a lineup or do an add/drop in the browser.")
        print("→ Watching for write requests for 180s (Ctrl-C to stop early)...")
        try:
            for _ in range(180):
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        browser.close()

    out = settings.data_dir / "captured_writes.json"
    out.write_text(json.dumps(captured, indent=2))
    print(f"\n✓ Captured {len(captured)} write request(s) -> {out}")
    if captured:
        print("Next: inspect the URL/headers/body to build a replay executor or confirm DOM-only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
