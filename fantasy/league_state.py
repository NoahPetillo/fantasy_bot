"""LeagueSnapshot — who owns whom right now, plus the free-agent pool.

The recommendation generators operate on a snapshot, not directly on ESPN, so the
same logic serves two sources:

- LIVE: built from the ESPN read client (rosters + free agents) — needs cookies.
- DRY-RUN: synthesized from the projection board via a snake draft, so the whole
  Phase-2 loop is demonstrable on historical data with no league connected.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fantasy.league_settings import LeagueSettings


@dataclass
class LeagueSnapshot:
    season: int
    week: int
    my_team_id: int
    teams: dict[int, list[str]]  # team_id -> [player_id]
    free_agents: list[str]  # [player_id]
    names: dict[str, str] = field(default_factory=dict)  # player_id -> display name
    positions: dict[str, str] = field(default_factory=dict)  # player_id -> position
    faab_remaining: dict[int, int] = field(default_factory=dict)
    team_names: dict[int, str] = field(default_factory=dict)

    def roster(self, team_id: int) -> list[str]:
        return self.teams.get(team_id, [])

    def my_roster(self) -> list[str]:
        return self.roster(self.my_team_id)

    def opponents(self) -> list[int]:
        return [t for t in self.teams if t != self.my_team_id]


def build_dryrun_snapshot(
    board: pd.DataFrame, league: LeagueSettings, season: int, week: int,
    my_team_id: int = 1, draft_noise: float = 0.4, seed: int = 7,
) -> LeagueSnapshot:
    """Synthesize a realistic league by snake-drafting the value board.

    Rosters fill by *noisy* best-available VOR (snake order) — ``draft_noise``
    perturbs the draft like real ADP uncertainty, so some high-value players slip
    to the free-agent pool (mid-season breakouts/returns), giving the waiver logic
    real upgrades to find. The next tier becomes free agents. Produces a coherent
    state for exercising start/sit, waivers, and trades.
    """
    n = league.team_count
    roster_size = league.roster.total_starters + league.roster.bench_size
    work = board.copy()
    sigma = draft_noise * float(work["vor"].std() or 1.0)
    rng = np.random.default_rng(seed)
    work["draft_key"] = work["vor"] + rng.normal(0.0, sigma, size=len(work))
    ranked = work.sort_values("draft_key", ascending=False).reset_index(drop=True)

    teams: dict[int, list[str]] = {t: [] for t in range(1, n + 1)}
    names, positions = {}, {}
    drafted = set()

    order = list(range(1, n + 1))
    pick_seq = []
    for rnd in range(roster_size):
        pick_seq.extend(order if rnd % 2 == 0 else order[::-1])

    it = ranked.itertuples(index=False)
    for team_id in pick_seq:
        for row in it:
            if row.player_id in drafted:
                continue
            teams[team_id].append(row.player_id)
            names[row.player_id] = row.player_display_name
            positions[row.player_id] = row.position
            drafted.add(row.player_id)
            break

    # Free agents: next best ~200 undrafted.
    fas = []
    for row in ranked.itertuples(index=False):
        if row.player_id not in drafted:
            fas.append(row.player_id)
            names[row.player_id] = row.player_display_name
            positions[row.player_id] = row.position
            if len(fas) >= 200:
                break

    return LeagueSnapshot(
        season=season, week=week, my_team_id=my_team_id, teams=teams, free_agents=fas,
        names=names, positions=positions,
        faab_remaining={t: league.faab_budget for t in teams},
        team_names={t: ("MY TEAM" if t == my_team_id else f"Team {t}") for t in teams},
    )


def build_live_snapshot(client, league: LeagueSettings, season: int, week: int):
    """Build a snapshot from the live ESPN league (needs cookies).

    Maps ESPN player ids -> gsis ids via the ff crosswalk so rosters align with
    the projection board (which is keyed by gsis id). Players without a crosswalk
    entry (many K/DST, some rookies) are kept by name but won't have projections.
    """
    from fantasy.config import settings as app_settings
    from fantasy.data.nfl import load_player_ids

    xwalk = load_player_ids()
    cols = {c.lower(): c for c in xwalk.columns}
    espn_c, gsis_c = cols.get("espn_id"), cols.get("gsis_id")
    espn_to_gsis = {}
    if espn_c and gsis_c:
        m = xwalk[[espn_c, gsis_c]].dropna()
        espn_to_gsis = {str(int(e)): g for e, g in zip(m[espn_c], m[gsis_c]) if str(e).strip()}

    def pid_of(p) -> str:
        espn_id = str(getattr(p, "playerId", "") or "")
        return espn_to_gsis.get(espn_id, f"espn:{espn_id}")

    teams, names, positions, team_names, faab = {}, {}, {}, {}, {}
    for t in client.teams():
        tid = getattr(t, "team_id", None)
        roster = getattr(t, "roster", []) or []
        teams[tid] = []
        for p in roster:
            pid = pid_of(p)
            teams[tid].append(pid)
            names[pid] = getattr(p, "name", "?")
            positions[pid] = getattr(p, "position", "?")
        team_names[tid] = getattr(t, "team_name", f"Team {tid}")
        # ESPN exposes remaining FAAB on the team in budget leagues.
        faab[tid] = int(getattr(t, "acquisition_budget_spent", 0) or 0)
        faab[tid] = league.faab_budget - faab[tid]

    fas = []
    for p in client.free_agents(size=300):
        pid = pid_of(p)
        fas.append(pid)
        names[pid] = getattr(p, "name", "?")
        positions[pid] = getattr(p, "position", "?")

    return LeagueSnapshot(
        season=season, week=week,
        my_team_id=app_settings.espn_team_id or (next(iter(teams)) if teams else 1),
        teams=teams, free_agents=fas, names=names, positions=positions,
        faab_remaining=faab or {t: league.faab_budget for t in teams}, team_names=team_names,
    )
