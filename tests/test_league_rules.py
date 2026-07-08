"""League rules persistence + merge layer (Phases 2-3 of the custom rules plan).

Covers: merge order (detected < overrides, per-key scoring, wholesale roster
slots), the rules API round-trip, cross-user isolation, staleness marking on
save, ESPN-failure fallback in `effective_settings`, override validation, and
that the catalog is served alongside the rules payload.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from fantasy.db.repos import add_league
from fantasy.league_rules import (
    RulesValidationError,
    effective_settings,
    merge_settings,
    save_overrides,
    settings_diff,
)


# ── merge_settings ──────────────────────────────────────────────────────────
def test_merge_order_detected_then_overrides():
    detected = {"team_count": 12, "scoring": {"receiving_yards": 0.1}}
    overrides = {"team_count": 10}
    merged = merge_settings(detected, overrides)
    assert merged.team_count == 10
    assert merged.scoring["receiving_yards"] == 0.1


def test_merge_scoring_is_per_key():
    detected = {"scoring": {"receiving_yards": 0.1, "receiving_tds": 6.0}}
    overrides = {"scoring": {"receiving_targets": 0.25}}
    merged = merge_settings(detected, overrides)
    assert merged.scoring == {
        "receiving_yards": 0.1, "receiving_tds": 6.0, "receiving_targets": 0.25,
    }


def test_merge_scoring_override_replaces_that_key_only():
    detected = {"scoring": {"receptions": 1.0}}
    overrides = {"scoring": {"receptions": 0.5}}
    merged = merge_settings(detected, overrides)
    assert merged.scoring["receptions"] == 0.5


def test_roster_slots_replace_wholesale():
    detected = {"roster": {"slots": {"QB": 1, "RB": 2, "WR": 2, "FLEX": 1, "K": 1}}}
    overrides = {"roster": {"slots": {"QB": 1, "RB": 2, "WR": 2, "FLEX": 2, "DP": 1, "HC": 1}}}
    merged = merge_settings(detected, overrides)
    # Wholesale replace: K from `detected` must NOT survive since overrides
    # submitted a full new dict without it.
    assert merged.roster.slots == {"QB": 1, "RB": 2, "WR": 2, "FLEX": 2, "DP": 1, "HC": 1}
    assert "K" not in merged.roster.slots


def test_merge_with_no_detected_and_no_overrides_is_default():
    merged = merge_settings(None, {})
    assert merged.team_count == 12  # LeagueSettings default


# ── settings_diff ────────────────────────────────────────────────────────────
def test_settings_diff_lists_only_differences():
    detected = {"scoring": {"receiving_targets": 0.0}, "roster": {"slots": {"FLEX": 1}}}
    overrides = {"scoring": {"receiving_targets": 0.25}, "roster": {"slots": {"FLEX": 1, "DP": 1}}}
    diff = settings_diff(detected, overrides)
    paths = {d["path"] for d in diff}
    assert "scoring.receiving_targets" in paths
    assert "roster.slots.DP" in paths
    assert "roster.slots.FLEX" not in paths  # same value in both -> not a diff


# ── DB-backed: save_overrides / effective_settings ──────────────────────────
def _mk_league(db, user):
    return add_league(db, user, espn_league_id=42, team_id=1, season=2026, name="Test League")


def test_save_overrides_persists_merges_and_marks_plan_stale(db):
    from fantasy.db.models import User

    user = User(clerk_user_id="u1", email="u1@ex.com")
    db.add(user)
    db.commit()
    db.refresh(user)
    lg = _mk_league(db, user)
    lg.settings_detected = {"team_count": 12, "scoring": {"receiving_yards": 0.1}}
    lg.draft_plan_built_at = datetime.now(timezone.utc)
    db.commit()

    merged = save_overrides(db, lg, {"scoring": {"receiving_targets": 0.25}})
    assert merged.scoring["receiving_targets"] == 0.25
    assert merged.scoring["receiving_yards"] == 0.1  # detected key preserved
    db.refresh(lg)
    assert lg.settings_overrides == {"scoring": {"receiving_targets": 0.25}}
    assert lg.draft_plan_built_at is None  # cleared as the staleness marker


def test_save_overrides_rejects_unknown_scoring_key(db):
    from fantasy.db.models import User

    user = User(clerk_user_id="u2", email="u2@ex.com")
    db.add(user)
    db.commit()
    db.refresh(user)
    lg = _mk_league(db, user)
    with pytest.raises(RulesValidationError):
        save_overrides(db, lg, {"scoring": {"not_a_real_stat": 1.0}})


def test_save_overrides_rejects_unknown_roster_slot(db):
    from fantasy.db.models import User

    user = User(clerk_user_id="u3", email="u3@ex.com")
    db.add(user)
    db.commit()
    db.refresh(user)
    lg = _mk_league(db, user)
    with pytest.raises(RulesValidationError):
        save_overrides(db, lg, {"roster": {"slots": {"NOT_A_SLOT": 1}}})


def test_effective_settings_falls_back_on_client_failure(db):
    from fantasy.db.models import User

    user = User(clerk_user_id="u4", email="u4@ex.com")
    db.add(user)
    db.commit()
    db.refresh(user)
    lg = _mk_league(db, user)
    lg.settings_detected = {"team_count": 12, "name": "Stored League"}
    db.commit()

    class ExplodingClient:
        def league_settings(self):
            raise RuntimeError("ESPN unreachable")

    ls = effective_settings(db, lg, ExplodingClient())
    assert ls.name == "Stored League"
    assert ls.team_count == 12


def test_effective_settings_no_client_uses_stored(db):
    from fantasy.db.models import User

    user = User(clerk_user_id="u5", email="u5@ex.com")
    db.add(user)
    db.commit()
    db.refresh(user)
    lg = _mk_league(db, user)
    lg.settings_detected = {"team_count": 14}
    lg.settings_overrides = {"scoring": {"receiving_targets": 0.25}}
    db.commit()

    ls = effective_settings(db, lg, None)
    assert ls.team_count == 14
    assert ls.scoring["receiving_targets"] == 0.25


def test_effective_settings_refreshes_from_client(db):
    from fantasy.db.models import User
    from fantasy.league_settings import LeagueSettings

    user = User(clerk_user_id="u6", email="u6@ex.com")
    db.add(user)
    db.commit()
    db.refresh(user)
    lg = _mk_league(db, user)

    class FakeClient:
        def league_settings(self):
            return LeagueSettings(team_count=10, name="Live League")

    ls = effective_settings(db, lg, FakeClient())
    assert ls.team_count == 10
    db.refresh(lg)
    assert lg.settings_detected is not None
    assert lg.settings_detected["team_count"] == 10
    assert lg.settings_updated_at is not None


# ── API: GET/PUT/refetch rules, ownership, catalog ──────────────────────────
def test_rules_get_returns_defaults_and_catalog(webapp):
    user = webapp.make_user("owner")
    lg = _mk_league(webapp.db, user)
    webapp.auth_as(user)
    r = webapp.client.get(f"/api/leagues/{lg.id}/rules")
    assert r.status_code == 200
    d = r.json()
    assert d["detected"] is None
    assert d["overrides"] == {}
    assert d["diff"] == []
    assert "scoring" in d["catalog"] and "roster_slots" in d["catalog"]
    assert d["merged"]["team_count"] == 12


def test_rules_put_get_round_trip(webapp):
    user = webapp.make_user("owner")
    lg = _mk_league(webapp.db, user)
    webapp.auth_as(user)

    put_body = {"overrides": {
        "scoring": {"receiving_targets": 0.25, "kickoff_return_yards": 0.25},
        "roster": {"slots": {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 2, "K": 1, "DP": 1, "HC": 1}},
    }}
    r = webapp.client.put(f"/api/leagues/{lg.id}/rules", json=put_body)
    assert r.status_code == 200
    put_d = r.json()
    assert put_d["ok"] is True
    assert put_d["stale"] == {"dashboard": True, "draft_plan": True}
    assert put_d["merged"]["scoring"]["receiving_targets"] == 0.25

    r2 = webapp.client.get(f"/api/leagues/{lg.id}/rules")
    d2 = r2.json()
    assert d2["overrides"]["scoring"]["receiving_targets"] == 0.25
    assert d2["merged"]["roster"]["slots"]["DP"] == 1
    assert d2["merged"]["roster"]["slots"]["HC"] == 1


def test_rules_put_rejects_unknown_scoring_key(webapp):
    user = webapp.make_user("owner")
    lg = _mk_league(webapp.db, user)
    webapp.auth_as(user)
    r = webapp.client.put(f"/api/leagues/{lg.id}/rules",
                          json={"overrides": {"scoring": {"not_a_key": 5.0}}})
    assert r.status_code == 400
    assert "detail" in r.json()


def test_rules_put_clears_draft_plan_stale_marker(webapp):
    user = webapp.make_user("owner")
    lg = _mk_league(webapp.db, user)
    lg.draft_plan_built_at = datetime.now(timezone.utc)
    webapp.db.commit()
    webapp.auth_as(user)

    webapp.client.put(f"/api/leagues/{lg.id}/rules", json={"overrides": {"team_count": 10}})
    webapp.db.expire_all()
    webapp.db.refresh(lg)
    assert lg.draft_plan_built_at is None


def test_rules_cross_user_access_404s(webapp):
    owner = webapp.make_user("owner")
    other = webapp.make_user("other")
    lg = _mk_league(webapp.db, owner)

    webapp.auth_as(other)
    r_get = webapp.client.get(f"/api/leagues/{lg.id}/rules")
    assert r_get.status_code == 404
    r_put = webapp.client.put(f"/api/leagues/{lg.id}/rules", json={"overrides": {}})
    assert r_put.status_code == 404
    r_refetch = webapp.client.post(f"/api/leagues/{lg.id}/rules/refetch")
    assert r_refetch.status_code == 404


def test_rules_refetch_without_espn_creds_errors(webapp):
    user = webapp.make_user("owner")
    lg = _mk_league(webapp.db, user)
    webapp.auth_as(user)
    r = webapp.client.post(f"/api/leagues/{lg.id}/rules/refetch")
    assert r.status_code == 502


# ── regressions: validation hardening (adversarial review findings) ─────────
def test_save_overrides_rejects_bad_scalars_without_persisting(db):
    """A payload pydantic would reject must 400 at validation time and leave
    nothing committed — a poisoned override layer used to brick GET + builds."""
    from fantasy.db.models import User

    user = User(clerk_user_id="u10", email="u10@ex.com")
    db.add(user)
    db.commit()
    db.refresh(user)
    lg = _mk_league(db, user)
    before = dict(lg.settings_overrides or {})

    for bad in (
        {"waiver_type": "bogus"},
        {"faab_budget": "unlimited"},
        {"playoff_weeks": "wk15"},
        {"is_dynasty": "maybe"},
        {"scoring_items_raw": {"1": 1.0}},   # not an allowed override field
        {"totally_unknown_field": 1},
    ):
        with pytest.raises(RulesValidationError):
            save_overrides(db, lg, bad)
    db.refresh(lg)
    assert (lg.settings_overrides or {}) == before


def test_save_overrides_caps_slot_counts_and_team_count(db):
    """No upper bound on slots/team_count let a crafted PUT queue an unbounded
    draft-plan build on the shared worker (DoS)."""
    from fantasy.db.models import User

    user = User(clerk_user_id="u11", email="u11@ex.com")
    db.add(user)
    db.commit()
    db.refresh(user)
    lg = _mk_league(db, user)

    with pytest.raises(RulesValidationError):
        save_overrides(db, lg, {"roster": {"slots": {"QB": 1, "BE": 10**9}}})
    with pytest.raises(RulesValidationError):
        save_overrides(db, lg, {"team_count": 64})
    with pytest.raises(RulesValidationError):
        save_overrides(db, lg, {"team_count": 1})
    # The documented maxima are accepted.
    merged = save_overrides(db, lg, {"team_count": 32,
                                     "roster": {"slots": {"QB": 1, "BE": 10}}})
    assert merged.team_count == 32


def test_save_overrides_rejects_non_finite_scoring(db):
    from fantasy.db.models import User

    user = User(clerk_user_id="u12", email="u12@ex.com")
    db.add(user)
    db.commit()
    db.refresh(user)
    lg = _mk_league(db, user)

    for bad_val in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(RulesValidationError):
            save_overrides(db, lg, {"scoring": {"receptions": bad_val}})


def test_effective_settings_timestamp_bumps_only_on_change(db):
    """Every dashboard build refreshes detected settings; the stale marker must
    only flip when ESPN actually reports something different. (The in-process
    fetch TTL is cleared between calls so each one really re-fetches.)"""
    from fantasy.db.models import User
    from fantasy.league_settings import LeagueSettings
    from fantasy import league_rules

    user = User(clerk_user_id="u13", email="u13@ex.com")
    db.add(user)
    db.commit()
    db.refresh(user)
    lg = _mk_league(db, user)

    class FixedClient:
        def league_settings(self):
            return LeagueSettings(team_count=10, name="Same League")

    league_rules._last_fetch.clear()
    effective_settings(db, lg, FixedClient())
    db.refresh(lg)
    first = lg.settings_updated_at
    assert first is not None

    league_rules._last_fetch.clear()
    effective_settings(db, lg, FixedClient())
    db.refresh(lg)
    assert lg.settings_updated_at == first  # unchanged settings -> no bump

    class ChangedClient:
        def league_settings(self):
            return LeagueSettings(team_count=14, name="Same League")

    league_rules._last_fetch.clear()
    effective_settings(db, lg, ChangedClient())
    db.refresh(lg)
    assert lg.settings_updated_at != first  # real change -> bump


def test_effective_settings_fetch_ttl_skips_back_to_back_refetches(db):
    """Two builds seconds apart must not pay two ESPN round-trips."""
    from fantasy.db.models import User
    from fantasy.league_settings import LeagueSettings
    from fantasy import league_rules

    user = User(clerk_user_id="u15", email="u15@ex.com")
    db.add(user)
    db.commit()
    db.refresh(user)
    lg = _mk_league(db, user)
    league_rules._last_fetch.clear()

    calls = {"n": 0}

    class CountingClient:
        def league_settings(self):
            calls["n"] += 1
            return LeagueSettings(team_count=10, name="TTL League")

    effective_settings(db, lg, CountingClient())
    effective_settings(db, lg, CountingClient())
    assert calls["n"] == 1  # second call within the TTL served without a fetch


def test_effective_settings_raises_on_auth_error(db):
    """EspnAuthError must PROPAGATE — the add-league flow validates cookies via
    the shell build, and the build-status path shows 'connect ESPN' from it."""
    from fantasy.db.models import User
    from fantasy.espn.client import EspnAuthError
    from fantasy import league_rules

    user = User(clerk_user_id="u14", email="u14@ex.com")
    db.add(user)
    db.commit()
    db.refresh(user)
    lg = _mk_league(db, user)
    league_rules._last_fetch.clear()

    class AuthFailClient:
        def league_settings(self):
            raise EspnAuthError("401")

    with pytest.raises(EspnAuthError):
        effective_settings(db, lg, AuthFailClient())
