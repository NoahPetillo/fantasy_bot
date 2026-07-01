"""Account-level ESPN checks (READ ONLY) for the connect flow.

``EspnClient`` is league-scoped; connecting an account happens *before* any league
is known, so validation here uses ESPN's public **fan** API to confirm a user's
``espn_s2``/``SWID`` actually authenticate, and to best-effort discover their
fantasy-football leagues for the add-league step. All requests are GETs — we never
write to ESPN. Cookies are never logged; errors are redacted.
"""

from __future__ import annotations

import logging

import requests

from fantasy.espn.client import USER_AGENT, EspnAuthError

log = logging.getLogger(__name__)

_FAN_API = "https://fan.api.espn.com/apis/v2/fans/{swid}"
# gameId / abbrev values ESPN uses for fantasy football.
_FFL_IDS = {"ffl", "1", 1}


def brace_swid(swid: str) -> str:
    """ESPN expects the SWID wrapped in braces; accept it with or without."""
    s = (swid or "").strip()
    if not s:
        return s
    if not s.startswith("{"):
        s = "{" + s
    if not s.endswith("}"):
        s = s + "}"
    return s


def _cookies(espn_s2: str, swid: str) -> dict[str, str]:
    return {"espn_s2": espn_s2, "SWID": brace_swid(swid)}


def fetch_fan_profile(espn_s2: str, swid: str) -> dict:
    """GET the fan profile for these cookies. Raises :class:`EspnAuthError` on
    401/403 (bad/expired cookies); ``requests.HTTPError`` on other HTTP errors."""
    resp = requests.get(
        _FAN_API.format(swid=brace_swid(swid)),
        params={"displayHiddenPrefs": "true", "source": "espncom-fantasy", "platform": "web"},
        cookies=_cookies(espn_s2, swid),
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        timeout=20,
    )
    if resp.status_code in (401, 403):
        raise EspnAuthError(f"ESPN returned {resp.status_code} — cookies invalid or expired.")
    resp.raise_for_status()
    return resp.json()


def validate_cookies(espn_s2: str, swid: str) -> bool:
    """True iff the cookies authenticate against ESPN. Auth failures → False;
    other network errors propagate (caller decides how to surface a transient
    failure rather than falsely reporting success)."""
    try:
        fetch_fan_profile(espn_s2, swid)
        return True
    except EspnAuthError:
        return False


def discover_ff_leagues(espn_s2: str, swid: str) -> list[dict]:
    """Best-effort list of the user's fantasy-football leagues
    ``[{league_id, team_id, season, name}]``. Never raises for parse issues —
    returns what it can (the reliable per-league validation happens at add-league)."""
    try:
        profile = fetch_fan_profile(espn_s2, swid)
    except (EspnAuthError, requests.RequestException, ValueError):
        return []
    out: list[dict] = []
    for pref in (profile.get("preferences") or []):
        try:
            entry = ((pref.get("metaData") or {}).get("entry") or {})
            game = entry.get("gameId") or entry.get("abbrev")
            if game is not None and game not in _FFL_IDS:
                continue
            season = entry.get("seasonId") or entry.get("season")
            team_id = entry.get("entryId") or entry.get("teamId")
            for grp in (entry.get("groups") or []):
                lid = grp.get("groupId") or grp.get("id")
                if lid is None:
                    continue
                out.append({
                    "league_id": int(lid),
                    "team_id": int(team_id) if team_id is not None else None,
                    "season": int(season) if season is not None else None,
                    "name": grp.get("groupName") or grp.get("name") or f"League {lid}",
                })
        except (TypeError, ValueError, KeyError):
            continue
    # De-dupe by league_id, keep first occurrence.
    seen, deduped = set(), []
    for lg in out:
        if lg["league_id"] not in seen:
            seen.add(lg["league_id"])
            deduped.append(lg)
    return deduped
