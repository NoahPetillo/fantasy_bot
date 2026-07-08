"""Static catalog of the ESPN scoring rules + roster slots users commonly
customize.

This is the single source of truth the rules UI renders its form from (grouped
sections, labels, and ESPN-default fallbacks) and the layer :mod:`fantasy.league_rules`
validates overrides against (an override key must name a stat this catalog knows
about; a roster slot must be one ESPN actually supports).

Canonical stat names match :data:`fantasy.espn.stat_ids.STATID_TO_CANONICAL`
exactly, so a catalog key can be looked up directly in ``LeagueSettings.scoring``.
Defaults here are ESPN's standard-scoring defaults, used only to pre-fill the
form when nothing has been detected/overridden yet — real valuation always reads
the league's actual detected/override values, never these.
"""

from __future__ import annotations

CATALOG: list[dict] = [
    # ── passing ──────────────────────────────────────────────────────────────
    {"key": "pass_attempts", "stat_id": 0, "label": "Pass attempt", "group": "passing", "default": 0.0},
    {"key": "pass_completions", "stat_id": 1, "label": "Completion", "group": "passing", "default": 0.0},
    {"key": "passing_yards", "stat_id": 3, "label": "Passing yard", "group": "passing", "default": 0.04},
    {"key": "passing_tds", "stat_id": 4, "label": "Passing TD", "group": "passing", "default": 4.0},
    {"key": "passing_2pt_conversions", "stat_id": 19, "label": "Passing 2pt conversion", "group": "passing", "default": 2.0},
    {"key": "passing_interceptions", "stat_id": 20, "label": "Interception thrown", "group": "passing", "default": -2.0},

    # ── rushing ──────────────────────────────────────────────────────────────
    {"key": "rushing_yards", "stat_id": 24, "label": "Rushing yard", "group": "rushing", "default": 0.1},
    {"key": "rushing_tds", "stat_id": 25, "label": "Rushing TD", "group": "rushing", "default": 6.0},
    {"key": "rushing_2pt_conversions", "stat_id": 26, "label": "Rushing 2pt conversion", "group": "rushing", "default": 2.0},

    # ── receiving ────────────────────────────────────────────────────────────
    {"key": "receiving_yards", "stat_id": 42, "label": "Receiving yard", "group": "receiving", "default": 0.1},
    {"key": "receptions", "stat_id": 53, "label": "Reception (PPR)", "group": "receiving", "default": 0.0},
    {"key": "receiving_targets", "stat_id": 58, "label": "Target", "group": "receiving", "default": 0.0},
    {"key": "receiving_tds", "stat_id": 43, "label": "Receiving TD", "group": "receiving", "default": 6.0},
    {"key": "receiving_2pt_conversions", "stat_id": 44, "label": "Receiving 2pt conversion", "group": "receiving", "default": 2.0},

    # ── returns (individual player return yardage; no D/ST in an IDP league) ──
    {"key": "kickoff_return_yards", "stat_id": 114, "label": "Kickoff return yard", "group": "returns", "default": 0.0},
    {"key": "punt_return_yards", "stat_id": 115, "label": "Punt return yard", "group": "returns", "default": 0.0},
    {"key": "kickoff_return_tds", "stat_id": 101, "label": "Kickoff return TD", "group": "returns", "default": 6.0},
    {"key": "punt_return_tds", "stat_id": 102, "label": "Punt return TD", "group": "returns", "default": 6.0},

    # ── IDP (individual defensive players) ──────────────────────────────────
    {"key": "def_tackles_solo", "stat_id": 108, "label": "Solo tackle", "group": "idp", "default": 0.0},
    {"key": "def_tackle_assists", "stat_id": 107, "label": "Assisted tackle", "group": "idp", "default": 0.0},
    {"key": "def_tackles_total", "stat_id": 109, "label": "Total tackles", "group": "idp", "default": 0.0},
    {"key": "dst_sacks", "stat_id": 99, "label": "Sacks (IDP/D-ST)", "group": "idp", "default": 0.0},
    {"key": "dst_interceptions", "stat_id": 95, "label": "Interception (IDP/D-ST)", "group": "idp", "default": 0.0},
    {"key": "def_fumbles_forced", "stat_id": 106, "label": "Fumble forced", "group": "idp", "default": 0.0},
    {"key": "dst_fumbles_recovered", "stat_id": 96, "label": "Fumble recovered (IDP/D-ST)", "group": "idp", "default": 0.0},
    {"key": "def_passes_defended", "stat_id": 113, "label": "Pass defended", "group": "idp", "default": 0.0},
    {"key": "def_tackles_for_loss", "stat_id": 112, "label": "Tackle for loss", "group": "idp", "default": 0.0},
    {"key": "dst_safeties", "stat_id": 98, "label": "Safety (IDP/D-ST)", "group": "idp", "default": 0.0},

    # ── head coach ───────────────────────────────────────────────────────────
    {"key": "hc_team_win", "stat_id": 155, "label": "Team win", "group": "hc", "default": 0.0},
    {"key": "hc_team_loss", "stat_id": 156, "label": "Team loss", "group": "hc", "default": 0.0},
    {"key": "hc_team_tie", "stat_id": 157, "label": "Team tie", "group": "hc", "default": 0.0},

    # ── kicking ──────────────────────────────────────────────────────────────
    {"key": "fg_made_50plus", "stat_id": 74, "label": "FG made 50+ yds", "group": "kicking", "default": 5.0},
    {"key": "fg_made_40_49", "stat_id": 77, "label": "FG made 40-49 yds", "group": "kicking", "default": 4.0},
    {"key": "fg_made_0_39", "stat_id": 80, "label": "FG made 0-39 yds", "group": "kicking", "default": 3.0},
    {"key": "fg_missed", "stat_id": 85, "label": "FG missed", "group": "kicking", "default": 0.0},
    {"key": "xp_made", "stat_id": 86, "label": "Extra point made", "group": "kicking", "default": 1.0},
    {"key": "xp_missed", "stat_id": 88, "label": "Extra point missed", "group": "kicking", "default": 0.0},

    # ── misc / fumbles / special teams TDs ──────────────────────────────────
    {"key": "fumbles_lost", "stat_id": 72, "label": "Fumble lost", "group": "misc", "default": -2.0},
    {"key": "fumble_recovery_tds", "stat_id": 68, "label": "Fumble recovery TD", "group": "misc", "default": 6.0},
    {"key": "interception_return_tds", "stat_id": 103, "label": "Interception return TD", "group": "misc", "default": 6.0},
    {"key": "fumble_return_tds", "stat_id": 104, "label": "Fumble return TD", "group": "misc", "default": 6.0},
]

# Canonical roster-slot display order (matches fantasy.espn.stat_ids.SLOT_IDS).
ROSTER_SLOT_ORDER: list[str] = [
    "QB", "TQB", "RB", "RB/WR", "WR", "WR/TE", "TE", "OP", "FLEX",
    "DP", "DL", "LB", "DB", "DT", "DE", "CB", "S", "ER",
    "D/ST", "K", "P", "HC", "BE", "IR",
]

# Fast lookup: canonical scoring key -> catalog entry.
CATALOG_BY_KEY: dict[str, dict] = {item["key"]: item for item in CATALOG}


def catalog_payload() -> dict:
    """Serializable form for the rules API/UI: catalog grouped implicitly by
    ``group`` (the frontend buckets by that field) plus the roster-slot order."""
    return {"scoring": CATALOG, "roster_slots": ROSTER_SLOT_ORDER}
