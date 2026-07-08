"""ESPN read client.

Two layers:

1. Raw v3 settings fetch (``requests`` + cookies) -> parsed into a fully
   league-adaptive :class:`fantasy.league_settings.LeagueSettings`. We parse the
   raw ``mSettings`` ourselves (rather than trusting a library to expose every
   field) so scoring rules, roster slots, waivers, and playoff weeks are read
   exactly as the league configures them.
2. ``espn-api`` (cwendt94) for the convenient object model: teams, rosters,
   box scores, free agents, draft, recent activity, transactions.

READ ONLY. There are no write methods here by design — execution lives behind
the approval gate in a separate, swappable module (Phase 3).
"""

from __future__ import annotations

import json
import logging
import time

import pandas as pd
import requests

from fantasy.config import settings as app_settings
from fantasy.espn.stat_ids import (
    IDP_SLOTS,
    PER_N_STATIDS,
    STATID_TO_CANONICAL,
    position_name,
    pro_team_abbr,
    slot_name,
)
from fantasy.league_settings import LeagueSettings, RosterRequirements, WaiverType

log = logging.getLogger(__name__)

READS_BASE = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
# ESPN receptions statId, used to detect TE-premium via pointsOverrides.
_RECEPTIONS_STAT_ID = 53
_TE_POSITION_ID = 4
# Season-projection cache TTL — preseason numbers move as ESPN updates them.
_SEASON_PROJ_TTL_SECONDS = 24 * 3600


class EspnAuthError(RuntimeError):
    pass


def _f(value) -> float | None:
    """Coerce to float, or None if unset/uncoercible (0.0 is kept)."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class EspnClient:
    def __init__(
        self,
        league_id: int | None = None,
        season: int | None = None,
        espn_s2: str | None = None,
        swid: str | None = None,
    ):
        self.league_id = league_id or app_settings.espn_league_id
        self.season = season or app_settings.espn_season
        self.espn_s2 = espn_s2 or app_settings.espn_s2
        self.swid = swid or app_settings.espn_swid_braced
        if not self.league_id:
            raise ValueError("No ESPN league_id configured (set ESPN_LEAGUE_ID).")
        self._league = None  # lazy espn-api League

    @property
    def cookies(self) -> dict[str, str]:
        c = {}
        if self.espn_s2:
            c["espn_s2"] = self.espn_s2
        if self.swid:
            c["SWID"] = self.swid
        return c

    # ── raw v3 endpoint ───────────────────────────────────────────────────────
    def _raw(self, views: list[str], historical: bool = False) -> dict:
        params = [("view", v) for v in views]
        if historical:
            url = f"{READS_BASE}/leagueHistory/{self.league_id}"
            params.append(("seasonId", str(self.season)))
        else:
            url = f"{READS_BASE}/seasons/{self.season}/segments/0/leagues/{self.league_id}"
        resp = requests.get(
            url,
            params=params,
            cookies=self.cookies,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            timeout=30,
        )
        if resp.status_code in (401, 403):
            raise EspnAuthError(
                f"ESPN returned {resp.status_code}. For a private league check espn_s2/SWID "
                f"are set and current; for an old season the data may have migrated to "
                f"leagueHistory (try historical=True)."
            )
        resp.raise_for_status()
        data = resp.json()
        # leagueHistory returns a list; current season returns an object.
        return data[0] if isinstance(data, list) else data

    # ── league settings -> fully adaptive LeagueSettings ──────────────────────
    def league_settings(self) -> LeagueSettings:
        """Read ``mSettings`` and build a LeagueSettings that drives all valuation."""
        try:
            data = self._raw(["mSettings"])
        except requests.HTTPError:
            log.info("mSettings via current endpoint failed; retrying leagueHistory.")
            data = self._raw(["mSettings"], historical=True)

        raw = data.get("settings", {})
        ls = LeagueSettings(
            league_id=self.league_id,
            season=self.season,
            name=raw.get("name"),
            team_count=data.get("size") or raw.get("size") or 12,
        )

        # Roster slots: lineupSlotCounts is {slotId: count}.
        slot_counts = raw.get("rosterSettings", {}).get("lineupSlotCounts", {})
        ls.roster = RosterRequirements(
            slots={slot_name(int(sid)): int(n) for sid, n in slot_counts.items() if int(n) > 0}
        )

        # Scoring: scoringItems is [{statId, points, pointsOverrides}].
        scoring_items = raw.get("scoringSettings", {}).get("scoringItems", [])
        ls.scoring, ls.scoring_items_raw, ls.position_reception_bonus = self._parse_scoring(
            scoring_items
        )

        # Acquisitions / waivers.
        acq = raw.get("acquisitionSettings", {})
        if acq.get("isUsingAcquisitionBudget"):
            ls.waiver_type = WaiverType.faab
            ls.faab_budget = int(acq.get("acquisitionBudget", 100))
        else:
            ls.waiver_type = WaiverType.rolling
        if acq.get("acquisitionLimit", -1) not in (-1, None):
            ls.acquisition_limit = int(acq["acquisitionLimit"])

        # Schedule / playoffs.
        sched = raw.get("scheduleSettings", {})
        ls.regular_season_weeks = int(sched.get("matchupPeriodCount", ls.regular_season_weeks))
        ls.playoff_team_count = int(raw.get("playoffTeamCount", ls.playoff_team_count))
        ls.playoff_weeks = self._infer_playoff_weeks(ls.regular_season_weeks, sched)

        # Format flags.
        draft = raw.get("draftSettings", {})
        ls.keeper_count = int(draft.get("keeperCount", 0) or 0)
        ls.is_dynasty = bool(draft.get("keeperCount", 0)) and draft.get("type") == "OFFLINE"
        ls.uses_idp = any(
            s in IDP_SLOTS and n > 0 for s, n in ls.roster.slots.items()
        )
        log.info("Loaded league settings: %s", ls.summary())
        return ls

    @staticmethod
    def _parse_scoring(
        scoring_items: list[dict],
    ) -> tuple[dict[str, float], dict[int, float], dict[str, float]]:
        canonical: dict[str, float] = {}
        per_n_extra: dict[str, float] = {}
        raw_map: dict[int, float] = {}
        reception_bonus: dict[str, float] = {}
        base_rec_points = 0.0
        unknown: list[int] = []
        for item in scoring_items:
            stat_id = int(item.get("statId", -1))
            pts = float(item.get("points", 0.0) or 0.0)
            raw_map[stat_id] = pts
            per_n = PER_N_STATIDS.get(stat_id)
            if per_n is not None:
                # "every N units" rule -> accumulated separately and added to the
                # per-unit value AFTER the loop, so the result doesn't depend on
                # whether ESPN lists the per-unit or per-N item first.
                name, n = per_n
                if pts:
                    per_n_extra[name] = per_n_extra.get(name, 0.0) + pts / n
                continue
            name = STATID_TO_CANONICAL.get(stat_id)
            if name is None:
                if pts:
                    unknown.append(stat_id)
                continue
            canonical[name] = pts
            overrides_raw = item.get("pointsOverrides", {}) or {}
            if overrides_raw and stat_id != _RECEPTIONS_STAT_ID:
                log.info(
                    "statId %s has position pointsOverrides %s (only the TE "
                    "reception premium is modeled; base points used otherwise)",
                    stat_id, overrides_raw,
                )
            if stat_id == _RECEPTIONS_STAT_ID:
                base_rec_points = pts
                overrides = item.get("pointsOverrides", {}) or {}
                te_override = overrides.get(str(_TE_POSITION_ID))
                if te_override is not None and float(te_override) != pts:
                    reception_bonus["TE"] = float(te_override) - base_rec_points
        for name, extra in per_n_extra.items():
            canonical[name] = canonical.get(name, 0.0) + extra
        if unknown:
            log.warning(
                "Unrecognized scoring statIds with nonzero points (verify in stat_ids.py): %s",
                sorted(set(unknown)),
            )
        return canonical, raw_map, reception_bonus

    @staticmethod
    def _infer_playoff_weeks(reg_weeks: int, sched: dict) -> list[int]:
        length = int(sched.get("playoffMatchupPeriodLength", 1) or 1)
        # Playoffs start the week after the regular season; ESPN weeks are 1-based.
        start = reg_weeks + 1
        # Typically a 3-round bracket; clamp to the 18-week NFL season.
        weeks = list(range(start, min(start + 3 * max(length, 1), 19)))
        return weeks or [reg_weeks + 1]

    # ── espn-api object model (lazy) ──────────────────────────────────────────
    def league(self):
        if self._league is None:
            from espn_api.football import League

            self._league = League(
                league_id=self.league_id,
                year=self.season,
                espn_s2=self.espn_s2,
                swid=self.swid,
            )
        return self._league

    def teams(self):
        return self.league().teams

    def my_team(self, team_id: int | None = None):
        tid = team_id or app_settings.espn_team_id
        if tid is None:
            return None
        for t in self.teams():
            if getattr(t, "team_id", None) == tid:
                return t
        return None

    def free_agents(self, week: int | None = None, size: int = 200, position: str | None = None):
        return self.league().free_agents(week=week, size=size, position=position)

    def box_scores(self, week: int | None = None):
        return self.league().box_scores(week)

    def draft(self):
        return self.league().draft

    def recent_activity(self, size: int = 25):
        return self.league().recent_activity(size=size)

    def transactions(self, scoring_period: int | None = None):
        return self.league().transactions(scoring_period)

    def week_projections(self, week: int, fa_size: int = 300) -> dict[str, float]:
        """ESPN's own projected points for ``week``, keyed by gsis player_id.

        Pulls rostered players from box scores and free agents from the FA list —
        these are the best projection source we have (they beat our model ~2%), so
        they become the primary input to the decision engine.
        """
        from fantasy.data.ids import crosswalk

        xw = crosswalk()
        out: dict[str, float] = {}

        def add(espn_id, proj):
            gid = xw.from_espn(espn_id)
            if gid and proj is not None:
                out[gid] = float(proj)

        try:
            for m in self.box_scores(week):
                for bp in (getattr(m, "home_lineup", []) or []) + (getattr(m, "away_lineup", []) or []):
                    add(getattr(bp, "playerId", None), getattr(bp, "projected_points", None))
        except Exception as e:  # noqa: BLE001
            log.warning("week_projections box_scores failed: %s", e)
        try:
            for p in self.free_agents(week=week, size=fa_size):
                stats = getattr(p, "stats", {}) or {}
                proj = (stats.get(week, {}) or {}).get("projected_points",
                                                       getattr(p, "projected_points", None))
                add(getattr(p, "playerId", None), proj)
        except Exception as e:  # noqa: BLE001
            log.warning("week_projections free_agents failed: %s", e)
        return out

    # ── season-long per-stat projections (kona_player_info) ───────────────────
    def season_stat_projections(self, refresh: bool = False) -> pd.DataFrame:
        """ESPN's season projection for every player as a per-stat frame.

        Reads the public ``kona_player_info`` view (works unauthenticated — these
        projections and draft ranks are public). For each player we keep the
        SEASON projection stat line (the ``stats`` entry with ``statSourceId==1``,
        ``statSplitTypeId==0`` and ``id == "10{season}"``), translated from ESPN
        statIds into our canonical stat columns (so ``receiving_targets`` etc. are
        available to the ScoringEngine), plus draft ranks + average draft position.

        Columns: ``espn_id, player_id (gsis|None), name, position, team, adp,
        auction_value, rank_ppr, rank_std`` and one column per canonical stat that
        ESPN projects. Cached to ``espn_season_proj_{season}.parquet``. Returns a
        typed empty frame if projections for ``season`` are not yet published.
        """
        path = app_settings.cache_dir / f"espn_season_proj_{self.season}.parquet"
        # Preseason projections shift as sources update, so the cache has a TTL;
        # a stale cache is still the fallback if the refetch fails or comes back
        # empty (never cache an empty frame — it would mask later publication).
        cache_fresh = (
            path.exists()
            and (time.time() - path.stat().st_mtime) < _SEASON_PROJ_TTL_SECONDS
        )
        if not refresh and cache_fresh:
            return pd.read_parquet(path)

        try:
            raw = self._kona_players()
        except Exception as e:  # noqa: BLE001 — stale cache beats no data
            if path.exists():
                log.warning("kona fetch failed (%s); serving stale cache.", e)
                return pd.read_parquet(path)
            raise
        if not raw:
            if path.exists():
                log.info("kona returned no players; serving stale cache.")
                return pd.read_parquet(path)
            return self._empty_season_proj()

        from fantasy.data.ids import crosswalk

        xw = crosswalk()
        proj_id = f"10{self.season}"
        rows: list[dict] = []
        projected = 0
        for entry in raw:
            p = entry.get("player") or {}
            espn_id = p.get("id")
            row: dict = {
                "espn_id": None if espn_id is None else str(espn_id),
                "player_id": xw.from_espn(espn_id) if espn_id is not None else None,
                "name": p.get("fullName"),
                "position": position_name(int(p.get("defaultPositionId", -1) or -1)),
                "team": pro_team_abbr(p.get("proTeamId")),
            }
            own = p.get("ownership") or {}
            row["adp"] = _f(own.get("averageDraftPosition"))
            row["auction_value"] = _f(own.get("auctionValueAverage"))
            ranks = p.get("draftRanksByRankType") or {}
            row["rank_ppr"] = (ranks.get("PPR") or {}).get("rank")
            row["rank_std"] = (ranks.get("STANDARD") or {}).get("rank")
            # Season projection stat line.
            stat_line = None
            for s in p.get("stats") or []:
                if (s.get("statSourceId") == 1 and s.get("statSplitTypeId") == 0
                        and str(s.get("id")) == proj_id):
                    stat_line = s.get("stats") or {}
                    break
            if stat_line:
                projected += 1
                for sid_str, value in stat_line.items():
                    try:
                        sid = int(sid_str)
                    except (TypeError, ValueError):
                        continue
                    canonical = STATID_TO_CANONICAL.get(sid)
                    if canonical is not None and value is not None:
                        row[canonical] = row.get(canonical, 0.0) + float(value)
            rows.append(row)

        df = pd.DataFrame(rows)
        if projected == 0:
            log.info("ESPN season projections for %s not published yet (id %s absent).",
                     self.season, proj_id)
        else:
            log.info("ESPN season projections: %d/%d players carry a %s stat line.",
                     projected, len(df), proj_id)
        df.to_parquet(path, index=False)
        return df

    def _kona_players(self, limit: int = 1500) -> list[dict]:
        """Fetch the raw kona_player_info list; empty on failure."""
        filt = json.dumps({
            "players": {
                "limit": limit,
                "sortDraftRanks": {"sortPriority": 100, "sortAsc": True, "value": "PPR"},
            }
        })
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json",
                   "X-Fantasy-Filter": filt}
        base = f"{READS_BASE}/seasons/{self.season}"
        # Prefer the real league (uses the league's own scoring for ranks); fall
        # back to the league-free defaults so it works before cookies/league exist.
        urls = [
            f"{base}/segments/0/leagues/{self.league_id}?view=kona_player_info",
            f"{base}/segments/0/leaguedefaults/3?view=kona_player_info",
        ]
        for url in urls:
            try:
                resp = requests.get(url, cookies=self.cookies, headers=headers, timeout=30)
                resp.raise_for_status()
                players = resp.json().get("players") or []
                if players:
                    return players
            except Exception as e:  # noqa: BLE001
                log.info("kona_player_info via %s failed: %s", url.split("/ffl")[-1][:50], e)
        return []

    @staticmethod
    def _empty_season_proj() -> pd.DataFrame:
        cols = ["espn_id", "player_id", "name", "position", "team", "adp",
                "auction_value", "rank_ppr", "rank_std"]
        return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})
