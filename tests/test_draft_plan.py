"""Draft Plan — computed offline from a synthetic season board (Phase 5).

The plan is a pure function of the merged league settings + the season value
board, so these tests build a fixed synthetic board and assert on structure, the
research-derived gates (no QB rounds 1-5 in 1-QB, K/DP last two rounds, HC last
only), that rules_impact only lists ACTIVE rules, the returner overlay surfaces,
and determinism. The API tests use the webapp harness with the season-board /
plan build monkeypatched instant so the background thread finishes quickly.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from fantasy.draft.plan import build_draft_plan
from fantasy.league_settings import LeagueSettings, RosterRequirements

# ── the user's 2026 league ──────────────────────────────────────────────────
_SCORING = {
    "passing_yards": 0.04, "passing_tds": 4.0, "passing_interceptions": -2.0,
    "rushing_yards": 0.1, "rushing_tds": 6.0,
    "receiving_yards": 0.1, "receiving_tds": 6.0,
    "receptions": 0.5, "receiving_targets": 0.25,
    "kickoff_return_yards": 0.25, "punt_return_yards": 0.25,
    "def_tackles_solo": 1.0, "def_tackle_assists": 0.5, "dst_sacks": 2.0,
    "hc_team_win": 5.0, "hc_team_loss": -5.0,
}
_ROSTER = RosterRequirements(
    slots={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 2, "K": 1, "DP": 1, "HC": 1, "BE": 7}
)


def _league() -> LeagueSettings:
    return LeagueSettings(league_id=1, season=2026, team_count=12,
                          scoring=dict(_SCORING), roster=_ROSTER)


def _plain_ppr_league() -> LeagueSettings:
    return LeagueSettings(
        league_id=2, season=2026, team_count=12,
        scoring={"passing_yards": 0.04, "passing_tds": 4.0, "rushing_yards": 0.1,
                 "rushing_tds": 6.0, "receiving_yards": 0.1, "receiving_tds": 6.0,
                 "receptions": 1.0},
        roster=RosterRequirements(
            slots={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "K": 1, "D/ST": 1, "BE": 6}),
    )


def _synthetic_board() -> pd.DataFrame:
    """~80 players across QB/RB/WR/TE/K/LB/S/DE + HC rows + two high-return WRs.

    proj/vor are laid out so the tiers are clean and deterministic: RB/WR are the
    deepest and most valuable, QBs are cheap (replacement-level), K/DP flat, HC
    small EV. ADP tracks proj rank so survival gating behaves sensibly.
    """
    rows = []
    pick = 1

    def add(pid, name, pos, proj, ret=0.0):
        nonlocal pick
        rows.append({"player_id": pid, "name": name, "position": pos,
                     "team": "AAA", "proj": float(proj), "return_pts": float(ret),
                     "adp": float(pick), "adp_sd": 6.0})
        pick += 1

    # RBs — steep top, long tail (24).
    for i in range(24):
        add(f"RB{i}", f"RB {i}", "RB", 320 - i * 8)
    # WRs — two of them carry big return value (edge). (26)
    for i in range(26):
        ret = 300.0 if i == 4 else (250.0 if i == 9 else 0.0)
        add(f"WR{i}", f"WR {i}", "WR", 315 - i * 7, ret=ret)
    # TEs — one elite, then a cliff (10).
    te_proj = [260, 200, 150, 140, 135, 130, 128, 125, 122, 120]
    for i, p in enumerate(te_proj):
        add(f"TE{i}", f"TE {i}", "TE", p)
    # QBs — cheap/flat, all clustered (10).
    for i in range(10):
        add(f"QB{i}", f"QB {i}", "QB", 300 - i * 3)
    # Kickers — flat (6).
    for i in range(6):
        add(f"K{i}", f"K {i}", "K", 140 - i * 2)
    # IDP: LBs dominate, plus a couple S/DE (16).
    for i in range(10):
        add(f"LB{i}", f"LB {i}", "LB", 160 - i * 3)
    for i in range(3):
        add(f"S{i}", f"S {i}", "S", 130 - i * 4)
    for i in range(3):
        add(f"DE{i}", f"DE {i}", "DE", 120 - i * 4)
    # HC rows (synthetic ids), small EV spread.
    for i, team in enumerate(["KC", "BUF", "PHI", "SF"]):
        rows.append({"player_id": f"HC:{team}", "name": f"HC {team}", "position": "HC",
                     "team": team, "proj": 55 - i * 5, "return_pts": 0.0,
                     "adp": 240.0, "adp_sd": 20.0})

    df = pd.DataFrame(rows)
    df["vor"] = df["proj"]  # simple monotone vor for deterministic tiering
    return df


@pytest.fixture
def board():
    return _synthetic_board()


@pytest.fixture(autouse=True)
def _no_returner_network(monkeypatch):
    """Keep the returner-role lookup offline (no depth-chart fetch)."""
    monkeypatch.setattr("fantasy.data.returns.current_returners",
                        lambda season, refresh=False: pd.DataFrame(
                            columns=["gsis_id", "name", "team", "role", "rank", "stale_season"]))


# ── structure ───────────────────────────────────────────────────────────────
def test_plan_has_all_sections(board):
    plan = build_draft_plan(_league(), 2026, board, my_slot=6)
    for key in ("league_summary", "rules_impact", "positional_value",
                "round_plan", "streamability", "returner_watch"):
        assert key in plan
    assert isinstance(plan["league_summary"], str) and plan["league_summary"]
    assert isinstance(plan["rules_impact"], list) and plan["rules_impact"]
    assert isinstance(plan["round_plan"], list) and plan["round_plan"]


def test_league_summary_mentions_custom_rules(board):
    summary = build_draft_plan(_league(), 2026, board, my_slot=6)["league_summary"]
    assert "12-team" in summary
    assert "target" in summary.lower()
    assert "return" in summary.lower()
    assert "coach" in summary.lower()


def test_round_plan_length_matches_starters_plus_bench(board):
    plan = build_draft_plan(_league(), 2026, board, my_slot=6)
    total = _league().roster.total_starters + _league().roster.bench_size
    assert len(plan["round_plan"]) == total


def test_positional_value_has_core_and_special_pools(board):
    pv = build_draft_plan(_league(), 2026, board, my_slot=6)["positional_value"]
    positions = {r["position"] for r in pv}
    assert {"QB", "RB", "WR", "TE", "K"} <= positions
    assert "DP" in positions and "HC" in positions
    for r in pv:
        for k in ("replacement_rank", "replacement_pts", "top3_avg_pts",
                  "dropoff_pts", "tier_note"):
            assert k in r


# ── gates ───────────────────────────────────────────────────────────────────
def test_qb_never_a_priority_in_rounds_1_to_5(board):
    plan = build_draft_plan(_league(), 2026, board, my_slot=6)
    for entry in plan["round_plan"]:
        if entry["round"] <= 5:
            prio_positions = {p["position"] for p in entry["priorities"]}
            assert "QB" not in prio_positions, f"QB in round {entry['round']}"
            assert "QB" in entry["avoid"]


def test_qb_allowed_from_round_6_in_superflex():
    sf = _league().model_copy(deep=True)
    slots = dict(sf.roster.slots)
    slots["OP"] = 1  # add a superflex slot
    sf.roster.slots = slots
    plan = build_draft_plan(sf, 2026, _synthetic_board(), my_slot=6)
    # In superflex QB is never gated out.
    for entry in plan["round_plan"]:
        assert "QB" not in entry["avoid"]


def test_k_and_dp_only_in_last_two_rounds(board):
    plan = build_draft_plan(_league(), 2026, board, my_slot=6)
    total = len(plan["round_plan"])
    for entry in plan["round_plan"]:
        prio = {p["position"] for p in entry["priorities"]}
        rounds_left = total - entry["round"] + 1
        if rounds_left > 2:
            assert "K" not in prio, f"K in round {entry['round']}"
            assert "DP" not in prio, f"DP in round {entry['round']}"
            assert "K" in entry["avoid"] and "DP" in entry["avoid"]


def test_hc_only_in_last_round(board):
    plan = build_draft_plan(_league(), 2026, board, my_slot=6)
    total = len(plan["round_plan"])
    for entry in plan["round_plan"]:
        prio = {p["position"] for p in entry["priorities"]}
        if "HC" in prio:
            assert entry["round"] == total, f"HC in round {entry['round']}, not last"


def test_pick_overall_snakes_correctly(board):
    plan = build_draft_plan(_league(), 2026, board, my_slot=3)
    # 12-team snake from slot 3: R1 pick 3, R2 pick 22 (12*2-3+1), R3 pick 27.
    picks = {e["round"]: e["pick_overall"] for e in plan["round_plan"]}
    assert picks[1] == 3
    assert picks[2] == 22
    assert picks[3] == 27


def test_default_slot_is_mid(board):
    plan = build_draft_plan(_league(), 2026, board, my_slot=None)
    # team_count // 2 == 6 -> R1 pick 6.
    assert plan["round_plan"][0]["pick_overall"] == 6


# ── returner watch ──────────────────────────────────────────────────────────
def test_returner_watch_present_and_lists_synthetic_returners(board):
    watch = build_draft_plan(_league(), 2026, board, my_slot=6)["returner_watch"]
    assert watch is not None
    names = {p["name"] for p in watch["players"]}
    assert "WR 4" in names and "WR 9" in names  # the two high-return WRs
    assert watch["players"][0]["return_pts"] >= watch["players"][-1]["return_pts"]


def test_returner_watch_absent_without_return_scoring():
    plan = build_draft_plan(_plain_ppr_league(), 2026, _synthetic_board(), my_slot=6)
    assert plan["returner_watch"] is None


# ── rules impact skips inactive rules ───────────────────────────────────────
def test_rules_impact_lists_custom_rules(board):
    impact = build_draft_plan(_league(), 2026, board, my_slot=6)["rules_impact"]
    rules = {e["rule"] for e in impact}
    assert "Points per target" in rules
    assert "Return yardage" in rules
    assert "Head coach (HC)" in rules
    assert any(r.endswith("FLEX") for r in rules)
    assert "Individual defensive player (DP)" in rules
    for e in impact:
        assert {"rule", "headline", "detail", "magnitude_pts"} <= set(e)


def test_rules_impact_skips_inactive_rules_in_plain_ppr():
    impact = build_draft_plan(_plain_ppr_league(), 2026, _synthetic_board())["rules_impact"]
    rules = {e["rule"] for e in impact}
    assert "Return yardage" not in rules
    assert "Individual defensive player (DP)" not in rules
    assert "Head coach (HC)" not in rules
    assert "Points per target" not in rules  # plain PPR scores receptions, not targets


# ── determinism ─────────────────────────────────────────────────────────────
def test_deterministic_given_fixed_board(board):
    a = build_draft_plan(_league(), 2026, board.copy(), my_slot=6)
    b = build_draft_plan(_league(), 2026, board.copy(), my_slot=6)
    assert a == b


# ── API: build thread + persistence + staleness + ownership ─────────────────
def _mk_league(db, user):
    from fantasy.db.repos import add_league

    return add_league(db, user, espn_league_id=77, team_id=1, season=2026, name="Plan League")


def _patch_instant_build(monkeypatch):
    """Make the plan-build thread instant + offline (no ESPN, no season board)."""
    import fantasy.api.app as api

    monkeypatch.setattr(api, "build_client_for_user",
                        lambda db, user, lid, season: (_ for _ in ()).throw(RuntimeError("no espn")))
    monkeypatch.setattr(api, "effective_settings",
                        lambda db, lg, client=None: _league())
    monkeypatch.setattr("fantasy.draft.season_board.build_season_board",
                        lambda season, league, refresh=False: _synthetic_board())


def _wait_plan_done(webapp, lg_id, timeout=8.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = webapp.client.get(f"/api/leagues/{lg_id}/draft-plan")
        if r.json()["status"] != "building":
            return r.json()
        time.sleep(0.05)
    raise AssertionError("plan build did not finish in time")


def test_api_build_then_get_persists_plan(webapp, monkeypatch):
    _patch_instant_build(monkeypatch)
    user = webapp.make_user("owner")
    lg = _mk_league(webapp.db, user)
    webapp.auth_as(user)

    r = webapp.client.post(f"/api/leagues/{lg.id}/draft-plan/build")
    assert r.status_code == 200 and r.json()["status"] == "building"

    done = _wait_plan_done(webapp, lg.id)
    assert done["status"] == "done"
    assert done["plan"] is not None
    assert "round_plan" in done["plan"]
    assert done["built_at"] is not None

    # Persisted on the league row.
    webapp.db.expire_all()
    webapp.db.refresh(lg)
    assert lg.draft_plan is not None
    assert lg.draft_plan_built_at is not None


def test_api_get_none_before_build(webapp):
    user = webapp.make_user("owner")
    lg = _mk_league(webapp.db, user)
    webapp.auth_as(user)
    r = webapp.client.get(f"/api/leagues/{lg.id}/draft-plan")
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "none"
    assert d["plan"] is None
    assert d["stale"] is False


def test_api_plan_goes_stale_after_rules_change(webapp, monkeypatch):
    _patch_instant_build(monkeypatch)
    user = webapp.make_user("owner")
    lg = _mk_league(webapp.db, user)
    webapp.auth_as(user)

    webapp.client.post(f"/api/leagues/{lg.id}/draft-plan/build")
    _wait_plan_done(webapp, lg.id)

    # A rules PUT clears draft_plan_built_at -> stale.
    webapp.client.put(f"/api/leagues/{lg.id}/rules", json={"overrides": {"team_count": 10}})
    d = webapp.client.get(f"/api/leagues/{lg.id}/draft-plan").json()
    assert d["stale"] is True
    assert d["plan"] is not None  # plan JSON still served, just flagged stale


def test_api_plan_stale_when_settings_updated_after_build(webapp, monkeypatch):
    _patch_instant_build(monkeypatch)
    user = webapp.make_user("owner")
    lg = _mk_league(webapp.db, user)
    webapp.auth_as(user)
    webapp.client.post(f"/api/leagues/{lg.id}/draft-plan/build")
    _wait_plan_done(webapp, lg.id)

    # Simulate a later ESPN settings refresh (settings_updated_at > built_at).
    webapp.db.refresh(lg)
    lg.settings_updated_at = (lg.draft_plan_built_at or datetime.now(timezone.utc)) + timedelta(hours=1)
    webapp.db.commit()
    d = webapp.client.get(f"/api/leagues/{lg.id}/draft-plan").json()
    assert d["stale"] is True


def test_api_draft_plan_cross_user_404s(webapp):
    owner = webapp.make_user("owner")
    other = webapp.make_user("other")
    lg = _mk_league(webapp.db, owner)
    webapp.auth_as(other)
    assert webapp.client.get(f"/api/leagues/{lg.id}/draft-plan").status_code == 404
    assert webapp.client.post(f"/api/leagues/{lg.id}/draft-plan/build").status_code == 404


# ── regressions: code-review findings ─────────────────────────────────────────
def test_no_hc_league_puts_dp_second_to_last_and_k_last(board):
    """Without an HC slot, DP and K must still each get a final-two-rounds pick:
    DP in the second-to-last round, K with the literal last pick."""
    from fantasy.league_settings import RosterRequirements

    league = _league()
    slots = dict(league.roster.slots)
    slots.pop("HC")
    league = league.model_copy(update={"roster": RosterRequirements(slots=slots)})
    scoring = dict(league.scoring)
    scoring.pop("hc_team_win"), scoring.pop("hc_team_loss")
    league = league.model_copy(update={"scoring": scoring})

    plan = build_draft_plan(league, 2026, board)
    rounds = plan["round_plan"]
    second_last, last = rounds[-2], rounds[-1]
    assert any(p["position"] == "DP" for p in second_last["priorities"])
    assert any(p["position"] == "K" for p in last["priorities"])
    assert not any(p["position"] == "HC" for r in rounds for p in r["priorities"])


def test_two_flex_impact_handles_non_generic_flex_slots():
    """A league running RB/WR + WR/TE (no slot literally named FLEX) must still
    compute a real replacement-rank shift, not '~0 RB and ~0 WR ranks'."""
    from fantasy.draft.plan import _impact_two_flex
    from fantasy.league_settings import RosterRequirements

    league = _league().model_copy(update={"roster": RosterRequirements(
        slots={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "RB/WR": 1, "WR/TE": 1,
               "K": 1, "BE": 6})})
    entry = _impact_two_flex(league)
    assert entry is not None
    assert "~0 RB and ~0 WR" not in entry["detail"]
