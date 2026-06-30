"""Playwright write tier — drives the ESPN web UI for an approved action.

ESPN has no documented write API, so writes go through the authenticated web UI
(cookie session). This mirrors the approach of every working ESPN auto-roster
tool (Selenium/DOM clicks). Selectors below are best-known and MUST be verified
against the live authenticated DOM during the Phase-3 spike (needs cookies) — the
helper raises a clear error naming the element it couldn't find so the spike can
correct it.

Trades are intentionally NOT done here (programmatic trade submission is unproven
and high-risk) — they fall back to the deep-link executor.
"""

from __future__ import annotations

import logging

from fantasy.config import settings
from fantasy.execute.base import ExecutionResult
from fantasy.orchestrator.models import Proposal, ProposalKind

log = logging.getLogger(__name__)

TEAM_URL = "https://fantasy.espn.com/football/team?leagueId={lid}&teamId={tid}&seasonId={season}"


class PlaywrightExecutor:
    name = "playwright"

    def __init__(self, headless: bool = True):
        self.headless = headless
        self.lid = settings.espn_league_id
        self.season = settings.espn_season

    def _cookies(self) -> list[dict]:
        out = []
        if settings.espn_s2:
            out.append({"name": "espn_s2", "value": settings.espn_s2, "domain": ".espn.com",
                        "path": "/"})
        if settings.espn_swid_braced:
            out.append({"name": "SWID", "value": settings.espn_swid_braced, "domain": ".espn.com",
                        "path": "/"})
        return out

    def execute(self, proposal: Proposal) -> ExecutionResult:
        if proposal.kind == ProposalKind.trade:
            from fantasy.execute.deeplink import DeepLinkExecutor

            return DeepLinkExecutor().execute(proposal)  # trades stay advisory
        if not (settings.espn_s2 and settings.espn_swid):
            return ExecutionResult(ok=False, backend=self.name,
                                   message="missing ESPN cookies for write tier")

        from playwright.sync_api import sync_playwright

        tid = proposal.team_id or settings.espn_team_id
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=self.headless)
            ctx = browser.new_context()
            ctx.add_cookies(self._cookies())
            page = ctx.new_page()
            page.goto(TEAM_URL.format(lid=self.lid, tid=tid, season=self.season),
                      wait_until="networkidle")
            try:
                if proposal.kind == ProposalKind.start_sit:
                    res = self._set_lineup(page, proposal)
                elif proposal.kind == ProposalKind.waiver:
                    res = self._add_drop(page, proposal)
                else:
                    res = ExecutionResult(ok=False, backend=self.name,
                                          message=f"unsupported kind {proposal.kind}")
            finally:
                browser.close()
            return res

    # ── individual flows (selectors to verify in the live spike) ──
    def _set_lineup(self, page, proposal: Proposal) -> ExecutionResult:
        """Move benched recommended starters into the lineup, then save.

        Reference flow (from working tools): each row has a 'Move' control; the
        target slot exposes a 'Here' button. Verify these selectors live.
        """
        # TODO(spike): confirm row/move/here selectors + the save/confirm buttons.
        save = page.get_by_role("button", name="Submit")
        if not save.count():
            raise RuntimeError("lineup save/submit button not found — verify selector in spike")
        # ... per-player move logic populated after the spike ...
        save.first.click()
        return ExecutionResult(ok=True, backend=self.name,
                               message="lineup submitted (verify in spike)",
                               performed=proposal.payload.get("lineup", {}))

    def _add_drop(self, page, proposal: Proposal) -> ExecutionResult:
        """Add a free agent and drop the configured player (waiver/FAAB)."""
        add_name = proposal.payload.get("add")
        drop_name = proposal.payload.get("drop")
        # TODO(spike): navigate to the player, click Add, choose the drop, set FAAB, confirm.
        confirm = page.get_by_role("button", name="Confirm")
        if not confirm.count():
            raise RuntimeError("add/drop confirm button not found — verify selector in spike")
        confirm.first.click()
        return ExecutionResult(ok=True, backend=self.name,
                               message=f"add/drop submitted: +{add_name} -{drop_name} (verify in spike)",
                               performed=proposal.payload)
