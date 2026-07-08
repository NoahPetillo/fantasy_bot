"""Persisted league rules: ESPN auto-detection + manual overrides, merged.

The 2026 league's rules aren't final on ESPN yet, so the app needs a rules layer
that works even when ``mSettings`` doesn't (yet) reflect reality: every league
setting is either **detected** (the last successful ESPN parse, cached) or
**overridden** (entered by hand in the rules UI), and the two are merged into one
:class:`~fantasy.league_settings.LeagueSettings` that flows into everything
downstream — scoring, VOR, lineup, waivers, trades, and the draft plan.

Merge order is always **detected < overrides**:

- ``scoring`` and ``position_reception_bonus`` merge **per key** (an override for
  one stat doesn't blow away the rest of the detected scoring map).
- ``roster.slots`` replaces **wholesale** when present in overrides (the rules UI
  always submits the complete slot dict, since a partial roster doesn't make
  sense — you can't "add one DP slot" without knowing the rest of the lineup).
- Every other field (``team_count``, ``waiver_type``, ``playoff_weeks``, ...) is a
  plain scalar/list replace when present in overrides.

Both layers persist as plain JSON on :class:`fantasy.db.models.League`
(``settings_detected`` / ``settings_overrides``) — never the merged result, so a
later change to either layer always recomputes correctly.
"""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timezone
from typing import Any

import pydantic
from sqlalchemy.orm import Session

from fantasy.db.models import League
from fantasy.espn.client import EspnAuthError
from fantasy.espn.rules_catalog import CATALOG_BY_KEY, ROSTER_SLOT_ORDER
from fantasy.league_settings import LeagueSettings, WaiverType

log = logging.getLogger(__name__)

# Fields that replace wholesale (JSON round-trip friendly: lists/scalars only).
_SCALAR_OVERRIDE_FIELDS = (
    "league_id", "season", "name", "team_count",
    "waiver_type", "faab_budget", "acquisition_limit",
    "regular_season_weeks", "playoff_team_count", "playoff_weeks", "matchup_periods",
    "keeper_count", "is_dynasty", "uses_idp",
)
# Everything a PUT may contain. Anything else is rejected up front so a typo'd
# or hostile key can never reach merge_settings/pydantic after being persisted.
_ALLOWED_OVERRIDE_KEYS = set(_SCALAR_OVERRIDE_FIELDS) | {
    "scoring", "position_reception_bonus", "roster",
}
# API-boundary bounds. The UI enforces the same limits, but the API is the trust
# boundary: an unbounded slot count would let one tenant queue a draft-plan build
# that loops for hours on the shared worker.
_MAX_SLOT_COUNT = 10
_MIN_TEAM_COUNT, _MAX_TEAM_COUNT = 2, 32
_INT_SCALAR_BOUNDS: dict[str, tuple[int, int]] = {
    "faab_budget": (0, 10_000),
    "acquisition_limit": (0, 1_000),
    "regular_season_weeks": (1, 18),
    "playoff_team_count": (0, 32),
    "matchup_periods": (1, 18),
    "keeper_count": (0, 32),
    "league_id": (0, 2**53),
    "season": (1990, 2100),
}
_BOOL_SCALARS = {"is_dynasty", "uses_idp"}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def merge_settings(detected: dict | None, overrides: dict | None) -> LeagueSettings:
    """Merge detected ESPN settings with manual overrides (overrides win).

    Both arguments are plain dicts shaped like ``LeagueSettings.model_dump()``
    (or a subset of it, for ``overrides``). Returns a fully validated
    :class:`LeagueSettings`.
    """
    merged: dict[str, Any] = dict(detected or {})
    overrides = overrides or {}

    # Scalars / lists: replace when present in overrides.
    for field in _SCALAR_OVERRIDE_FIELDS:
        if field in overrides:
            merged[field] = overrides[field]

    # scoring / scoring_items_raw: merge per-key.
    if "scoring" in overrides:
        scoring = dict(merged.get("scoring") or {})
        scoring.update(overrides["scoring"] or {})
        merged["scoring"] = scoring
    if "scoring_items_raw" in overrides:
        raw = dict(merged.get("scoring_items_raw") or {})
        raw.update(overrides["scoring_items_raw"] or {})
        merged["scoring_items_raw"] = raw

    # position_reception_bonus: merge per-key.
    if "position_reception_bonus" in overrides:
        bonus = dict(merged.get("position_reception_bonus") or {})
        bonus.update(overrides["position_reception_bonus"] or {})
        merged["position_reception_bonus"] = bonus

    # roster.slots: replace WHOLESALE when present.
    roster_override = overrides.get("roster") or {}
    if "slots" in roster_override:
        merged_roster = dict(merged.get("roster") or {})
        merged_roster["slots"] = dict(roster_override["slots"])
        merged["roster"] = merged_roster

    return LeagueSettings.model_validate(merged)


# Skip re-fetching mSettings when it was read this recently — a dashboard build
# plus a draft-plan build seconds apart shouldn't pay 2-3 ESPN round-trips on the
# shared worker. In-process only; the /rules/refetch endpoint always fetches.
_FETCH_TTL_SECONDS = 600
_last_fetch: dict[str, float] = {}  # str(league.id) -> monotonic time of last read


def effective_settings(db: Session, league: League, client=None) -> LeagueSettings:
    """The LeagueSettings that should drive every computation for this league.

    If ``client`` is given, refresh ``settings_detected`` from ESPN first.
    :class:`EspnAuthError` PROPAGATES — broken credentials must surface (the
    add-league flow and build status rely on it) — but any other failure falls
    back to the stored detected settings, logged, never raised, since a live
    rules layer must not block a build just because ESPN is briefly unreachable.
    Then merge with the stored overrides.
    """
    detected = league.settings_detected
    key = str(league.id)
    fetched_recently = (time.monotonic() - _last_fetch.get(key, float("-inf"))
                        ) < _FETCH_TTL_SECONDS
    if client is not None and not fetched_recently:
        try:
            ls = client.league_settings()
            detected = ls.model_dump(mode="json")
            _last_fetch[key] = time.monotonic()
            # Only bump the timestamp when ESPN actually reports something new —
            # otherwise every routine build would flag the draft plan as stale.
            if detected != league.settings_detected:
                league.settings_detected = detected
                league.settings_updated_at = _utcnow()
                db.commit()
        except EspnAuthError:
            raise
        except Exception as e:  # noqa: BLE001 — non-auth ESPN failures fall back
            log.warning(
                "league_settings() failed for league %s; falling back to stored "
                "settings_detected: %s", league.id, e,
            )
            db.rollback()
            detected = league.settings_detected

    overrides = league.settings_overrides or {}
    if detected is None and not overrides:
        return LeagueSettings()
    return merge_settings(detected, overrides)


class RulesValidationError(ValueError):
    """Raised when a rules override payload fails validation."""


def _require_number(val: Any, label: str) -> None:
    if not isinstance(val, (int, float)) or isinstance(val, bool):
        raise RulesValidationError(f"{label} must be a number")
    if not math.isfinite(float(val)):
        raise RulesValidationError(f"{label} must be a finite number")


def _validate_overrides(overrides: dict) -> None:
    unknown = set(overrides) - _ALLOWED_OVERRIDE_KEYS
    if unknown:
        raise RulesValidationError(f"Unknown override field(s): {sorted(unknown)}")

    scoring = overrides.get("scoring")
    if scoring is not None:
        if not isinstance(scoring, dict):
            raise RulesValidationError("scoring must be an object of {stat_key: points}")
        for key, val in scoring.items():
            if key not in CATALOG_BY_KEY:
                raise RulesValidationError(f"Unknown scoring key: {key!r}")
            _require_number(val, f"scoring[{key!r}]")

    bonus = overrides.get("position_reception_bonus")
    if bonus is not None:
        if not isinstance(bonus, dict):
            raise RulesValidationError("position_reception_bonus must be an object")
        for key, val in bonus.items():
            _require_number(val, f"position_reception_bonus[{key!r}]")

    roster = overrides.get("roster")
    if roster is not None:
        if not isinstance(roster, dict):
            raise RulesValidationError("roster must be an object")
        slots = roster.get("slots")
        if slots is not None:
            if not isinstance(slots, dict):
                raise RulesValidationError("roster.slots must be an object of {slot_name: count}")
            for name, count in slots.items():
                if name not in ROSTER_SLOT_ORDER:
                    raise RulesValidationError(f"Unknown roster slot: {name!r}")
                if (not isinstance(count, int) or isinstance(count, bool)
                        or not 0 <= count <= _MAX_SLOT_COUNT):
                    raise RulesValidationError(
                        f"roster.slots[{name!r}] must be an integer 0-{_MAX_SLOT_COUNT}")

    team_count = overrides.get("team_count")
    if team_count is not None and (
            not isinstance(team_count, int) or isinstance(team_count, bool)
            or not _MIN_TEAM_COUNT <= team_count <= _MAX_TEAM_COUNT):
        raise RulesValidationError(
            f"team_count must be an integer {_MIN_TEAM_COUNT}-{_MAX_TEAM_COUNT}")

    # Scalar fields (settable via API even though the UI doesn't expose them all).
    if "waiver_type" in overrides:
        valid = {w.value for w in WaiverType}
        if overrides["waiver_type"] not in valid:
            raise RulesValidationError(f"waiver_type must be one of {sorted(valid)}")
    if "name" in overrides and overrides["name"] is not None and not isinstance(overrides["name"], str):
        raise RulesValidationError("name must be a string")
    for field in _BOOL_SCALARS:
        if field in overrides and not isinstance(overrides[field], bool):
            raise RulesValidationError(f"{field} must be true/false")
    if "playoff_weeks" in overrides:
        weeks = overrides["playoff_weeks"]
        if (not isinstance(weeks, list) or
                any(not isinstance(w, int) or isinstance(w, bool) or not 1 <= w <= 18 for w in weeks)):
            raise RulesValidationError("playoff_weeks must be a list of week numbers 1-18")
    for field, (lo, hi) in _INT_SCALAR_BOUNDS.items():
        if field in overrides and overrides[field] is not None:
            val = overrides[field]
            if not isinstance(val, int) or isinstance(val, bool) or not lo <= val <= hi:
                raise RulesValidationError(f"{field} must be an integer {lo}-{hi}")


def save_overrides(db: Session, league: League, overrides: dict) -> LeagueSettings:
    """Validate + persist the override layer; clears the draft-plan staleness
    marker (a rules change always invalidates any cached draft plan) and returns
    the freshly merged settings.

    NOTE: the caller submits the COMPLETE override layer (everything that should
    differ from ESPN-detected) — the stored layer is replaced wholesale, and the
    merge is validated BEFORE anything is persisted, so a payload pydantic would
    reject can never be committed and brick the league."""
    _validate_overrides(overrides)
    try:
        merged = merge_settings(league.settings_detected, overrides)
    except pydantic.ValidationError as e:
        raise RulesValidationError(f"Overrides don't form valid league settings: {e}") from e
    league.settings_overrides = overrides
    league.draft_plan_built_at = None
    db.commit()
    db.refresh(league)
    return merged


def _get_in(d: dict, path: list[str]) -> Any:
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def settings_diff(detected: dict | None, overrides: dict) -> list[dict]:
    """List every field where an override differs from the detected value —
    drives the "ESPN: X" diff pills in the rules UI. Each entry is
    ``{path, detected, override}`` where ``path`` is dotted (e.g.
    ``scoring.receiving_targets`` or ``roster.slots.DP``)."""
    detected = detected or {}
    overrides = overrides or {}
    diffs: list[dict] = []

    for field in _SCALAR_OVERRIDE_FIELDS:
        if field in overrides and overrides[field] != detected.get(field):
            diffs.append({"path": field, "detected": detected.get(field), "override": overrides[field]})

    scoring_over = overrides.get("scoring") or {}
    for key, val in scoring_over.items():
        det_val = _get_in(detected, ["scoring", key])
        if val != det_val:
            diffs.append({"path": f"scoring.{key}", "detected": det_val, "override": val})

    bonus_over = overrides.get("position_reception_bonus") or {}
    for key, val in bonus_over.items():
        det_val = _get_in(detected, ["position_reception_bonus", key])
        if val != det_val:
            diffs.append({"path": f"position_reception_bonus.{key}", "detected": det_val, "override": val})

    slots_over = _get_in(overrides, ["roster", "slots"]) or {}
    for key, val in slots_over.items():
        det_val = _get_in(detected, ["roster", "slots", key])
        if val != det_val:
            diffs.append({"path": f"roster.slots.{key}", "detected": det_val, "override": val})

    return diffs
