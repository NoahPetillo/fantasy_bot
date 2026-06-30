"""Deep-link executor — the safe default.

Doesn't write to ESPN; it produces a one-tap URL to the exact ESPN screen plus an
instruction so the user completes the action manually. This is the recommended
mode for trades (programmatic trade POST is unproven) and a zero-risk option for
everything. "Executed" here means "hand-off link delivered".
"""

from __future__ import annotations

from fantasy.config import settings
from fantasy.execute.base import ExecutionResult
from fantasy.orchestrator.models import Proposal, ProposalKind

BASE = "https://fantasy.espn.com/football"


class DeepLinkExecutor:
    name = "deeplink"

    def execute(self, proposal: Proposal) -> ExecutionResult:
        lid = settings.espn_league_id
        tid = proposal.team_id or settings.espn_team_id
        season = settings.espn_season
        kind = proposal.kind

        if kind == ProposalKind.start_sit:
            url = f"{BASE}/team?leagueId={lid}&teamId={tid}&seasonId={season}"
            instr = "Open your roster and set the recommended lineup before kickoff."
        elif kind == ProposalKind.waiver:
            url = f"{BASE}/players/add?leagueId={lid}&teamId={tid}"
            add, drop = proposal.payload.get("add"), proposal.payload.get("drop")
            bid = proposal.payload.get("faab_bid")
            instr = f"Add the player and drop {drop}." + (f" Bid ${bid} FAAB." if bid else "")
        elif kind == ProposalKind.trade:
            other = proposal.payload.get("with_team")
            url = f"{BASE}/trade?leagueId={lid}&teamId={tid}"
            if other:
                url += f"&tradeTargetTeamId={other}"
            instr = "Open the trade screen, select the players, and send the offer."
        else:
            url = f"{BASE}/team?leagueId={lid}&teamId={tid}"
            instr = "Review in ESPN."

        return ExecutionResult(
            ok=True, backend=self.name, ref=url,
            message=f"Tap to complete in ESPN: {url}\n{instr}",
            performed={"url": url, "instruction": instr},
        )
