"""ESPN fantasy football ID maps and the bridge to nflverse stat columns.

ESPN encodes everything as integers: lineup slots (``slotCategoryId``), player
positions (``defaultPositionId``), and scoring rules (``scoringItems`` with a
``statId`` + ``points``). To stay fully league-adaptive we read those raw IDs and
translate them into *canonical* stat names that we can also compute from nflverse
historical data — so the SAME scoring rules score both projections and backtests.

NOTE: the offensive stat IDs below are high-confidence (used by the broader
community / the ``espn-api`` library). Kicker/DST/IDP IDs are best-effort and are
VERIFIED in Phase 0 against the real league's ``scoringItems`` — any ``statId``
we don't recognize is logged (never silently dropped) so the table can be
completed against the user's actual settings.
"""

from __future__ import annotations

# ── Lineup slot IDs (slotCategoryId) ──────────────────────────────────────────
SLOT_IDS: dict[int, str] = {
    0: "QB",
    1: "TQB",  # team QB
    2: "RB",
    3: "RB/WR",
    4: "WR",
    5: "WR/TE",
    6: "TE",
    7: "OP",  # superflex / offensive player
    8: "DT",
    9: "DE",
    10: "LB",
    11: "DL",
    12: "CB",
    13: "S",
    14: "DB",
    15: "DP",
    16: "D/ST",
    17: "K",
    18: "P",
    19: "HC",
    20: "BE",  # bench
    21: "IR",
    23: "FLEX",  # RB/WR/TE
    24: "ER",  # edge rusher
    25: "Rookie",
}

# Slots that hold a starter and therefore count toward replacement-level math.
STARTER_SLOTS: set[str] = {
    "QB", "TQB", "RB", "RB/WR", "WR", "WR/TE", "TE", "OP", "FLEX",
    "DT", "DE", "LB", "DL", "CB", "S", "DB", "DP", "D/ST", "K", "ER",
}
NON_STARTER_SLOTS: set[str] = {"BE", "IR"}

# Which player positions are eligible for each flex-style slot (for VOR / lineup LP).
FLEX_ELIGIBILITY: dict[str, set[str]] = {
    "FLEX": {"RB", "WR", "TE"},
    "RB/WR": {"RB", "WR"},
    "WR/TE": {"WR", "TE"},
    "OP": {"QB", "RB", "WR", "TE"},  # superflex
}

# ── Player position IDs (defaultPositionId) ───────────────────────────────────
POSITION_IDS: dict[int, str] = {
    1: "QB",
    2: "RB",
    3: "WR",
    4: "TE",
    5: "K",
    16: "D/ST",
    # IDP
    9: "DT", 10: "DE", 11: "LB", 12: "CB", 13: "S",
}

# ── Scoring stat IDs (statId) → canonical stat name ───────────────────────────
# Canonical names are chosen to align with nflverse columns where possible.
STATID_TO_CANONICAL: dict[int, str] = {
    # Passing
    0: "pass_attempts",
    1: "pass_completions",
    3: "passing_yards",
    4: "passing_tds",
    19: "passing_2pt_conversions",
    20: "passing_interceptions",
    # Rushing
    23: "rushing_attempts",
    24: "rushing_yards",
    25: "rushing_tds",
    26: "rushing_2pt_conversions",
    # Receiving
    42: "receiving_yards",
    43: "receiving_tds",
    44: "receiving_2pt_conversions",
    53: "receptions",
    58: "receiving_targets",
    # First downs (PPR-FD leagues)
    62: "passing_first_downs",
    63: "rushing_first_downs",
    64: "receiving_first_downs",
    # Fumbles
    68: "fumble_recovery_tds",
    72: "fumbles_lost",
    # Kicking (best-effort; verify Phase 0)
    74: "fg_made_50plus",
    77: "fg_made_40_49",
    80: "fg_made_0_39",
    85: "fg_missed",
    86: "xp_made",
    88: "xp_missed",
    # Team Defense / Special Teams (best-effort; verify Phase 0)
    89: "dst_points_allowed_0",
    90: "dst_points_allowed_1_6",
    91: "dst_points_allowed_7_13",
    92: "dst_points_allowed_14_17",
    93: "dst_blocked_kick",
    95: "dst_interceptions",
    96: "dst_fumbles_recovered",
    97: "dst_blocked_kick_for_td",
    98: "dst_safeties",
    99: "dst_sacks",
    101: "kickoff_return_tds",
    102: "punt_return_tds",
    103: "interception_return_tds",
    104: "fumble_return_tds",
    105: "dst_touchdowns",
}

# ── Canonical stat name → nflverse weekly column (for backtest/training) ───────
# Lets the ScoringEngine compute fantasy points from nflverse `load_player_stats`
# using the league's OWN scoring map. Column names are normalized in
# fantasy.data loaders; missing columns resolve to 0 with a one-time warning.
CANONICAL_TO_NFLVERSE: dict[str, str] = {
    "pass_attempts": "attempts",
    "pass_completions": "completions",
    "passing_yards": "passing_yards",
    "passing_tds": "passing_tds",
    "passing_2pt_conversions": "passing_2pt_conversions",
    "passing_interceptions": "passing_interceptions",
    "rushing_attempts": "carries",
    "rushing_yards": "rushing_yards",
    "rushing_tds": "rushing_tds",
    "rushing_2pt_conversions": "rushing_2pt_conversions",
    "receiving_yards": "receiving_yards",
    "receiving_tds": "receiving_tds",
    "receiving_2pt_conversions": "receiving_2pt_conversions",
    "receptions": "receptions",
    "receiving_targets": "targets",
    "passing_first_downs": "passing_first_downs",
    "rushing_first_downs": "rushing_first_downs",
    "receiving_first_downs": "receiving_first_downs",
    "fumbles_lost": "fumbles_lost",  # synthesized = sack+rushing+receiving fumbles lost
    "special_teams_tds": "special_teams_tds",
}


def slot_name(slot_id: int) -> str:
    return SLOT_IDS.get(slot_id, f"SLOT_{slot_id}")


def position_name(position_id: int) -> str:
    return POSITION_IDS.get(position_id, f"POS_{position_id}")
