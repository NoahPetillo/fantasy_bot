"""Deterministic data tools for the Q&A chatbot.

Each tool answers an atomic question from real data — nflverse weekly stats, our
projection board, the league's parsed scoring — and returns a short text result.
The LLM (in agent.py) composes them; it never produces a number itself, so answers
are grounded and verifiable. The same tools power the keyless fallback parser.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from fantasy.data.ids import crosswalk, norm_name
from fantasy.data.nfl import load_weekly
from fantasy.league_settings import LeagueSettings
from fantasy.valuation.scoring import ScoringEngine

log = logging.getLogger(__name__)

# Question stat keyword -> nflverse columns to sum.
STAT_ALIASES: dict[str, list[str]] = {
    "touchdowns": ["passing_tds", "rushing_tds", "receiving_tds"],
    "tds": ["passing_tds", "rushing_tds", "receiving_tds"],
    "td": ["passing_tds", "rushing_tds", "receiving_tds"],
    "total_tds": ["passing_tds", "rushing_tds", "receiving_tds"],
    "passing_tds": ["passing_tds"], "pass_tds": ["passing_tds"],
    "rushing_tds": ["rushing_tds"], "rush_tds": ["rushing_tds"],
    "receiving_tds": ["receiving_tds"], "rec_tds": ["receiving_tds"],
    "receptions": ["receptions"], "catches": ["receptions"], "rec": ["receptions"],
    "targets": ["targets"],
    "receiving_yards": ["receiving_yards"], "rec_yards": ["receiving_yards"], "rec_yds": ["receiving_yards"],
    "rushing_yards": ["rushing_yards"], "rush_yards": ["rushing_yards"], "rush_yds": ["rushing_yards"],
    "passing_yards": ["passing_yards"], "pass_yards": ["passing_yards"], "pass_yds": ["passing_yards"],
    "carries": ["carries"], "rush_attempts": ["carries"], "rushes": ["carries"],
    "attempts": ["attempts"], "pass_attempts": ["attempts"],
    "completions": ["completions"],
    "interceptions": ["passing_interceptions"], "ints": ["passing_interceptions"],
    "fumbles_lost": ["fumbles_lost"],
}
_SUPPORTED = "touchdowns, receptions, targets, rushing/receiving/passing yards, carries, completions, interceptions"


@dataclass
class ChatContext:
    """Everything the tools need for one league/season, assembled cheaply."""
    season: int
    board_index: dict = field(default_factory=dict)     # norm_name -> {name,pos,proj,vor}
    league_summary: str = ""
    scoring: dict = field(default_factory=dict)          # canonical stat -> points
    te_premium: dict = field(default_factory=dict)
    roster: dict = field(default_factory=dict)
    _cache: dict = field(default_factory=dict)

    @classmethod
    def from_snapshot(cls, snap: dict) -> "ChatContext":
        team = (snap or {}).get("team", {})
        ls = (snap or {}).get("league_settings", {})
        return cls(
            season=int(team.get("season") or 0) or _default_season(),
            board_index=(snap or {}).get("board_index", {}) or {},
            league_summary=ls.get("summary") or team.get("league", ""),
            scoring=ls.get("scoring", {}) or {},
            te_premium=ls.get("te_premium", {}) or {},
            roster=ls.get("roster", {}) or {},
        )

    def weekly(self, season: int | None = None) -> pd.DataFrame:
        s = season or self.season
        if s not in self._cache:
            try:
                df = load_weekly([s])
                self._cache[s] = df[df["season"] == s].copy() if "season" in df.columns else df
            except Exception as e:  # noqa: BLE001
                # Season data may not exist yet (offseason) or be unreachable — degrade
                # to empty so player lookups just miss instead of crashing the chat.
                logging.getLogger(__name__).warning("weekly stats for %s unavailable: %s", s, e)
                self._cache[s] = pd.DataFrame()
        return self._cache[s]

    def engine(self) -> ScoringEngine | None:
        if not self.scoring:
            return None
        ls = LeagueSettings(scoring=dict(self.scoring))
        ls.position_reception_bonus = dict(self.te_premium)
        return ScoringEngine(ls)


def _default_season() -> int:
    from fantasy.config import settings
    return settings.espn_season


# ── name resolution ───────────────────────────────────────────────────────────
# Words that are never a player name — so a single token like "how" or "score"
# can't accidentally match a real player (the old substring match did exactly that).
_STOP = {
    "how", "many", "much", "what", "whats", "who", "when", "where", "which", "does",
    "did", "do", "has", "have", "had", "is", "are", "was", "were", "the", "a", "an",
    "in", "on", "of", "to", "for", "since", "after", "before", "from", "until", "week",
    "weeks", "this", "that", "year", "season", "td", "tds", "touchdown", "touchdowns",
    "point", "points", "pts", "projected", "project", "projection", "yard", "yards",
    "reception", "receptions", "catch", "catches", "target", "targets", "score",
    "scored", "scores", "make", "made", "makes", "get", "got", "total", "and", "vs",
    "my", "league", "rushing", "receiving", "passing", "rush", "rec", "pass", "carry",
    "carries", "completion", "completions", "interception", "interceptions", "fantasy",
    "injured", "hurt", "out", "his", "her", "their", "they", "i", "you", "me",
}
_INDEX_CACHE: dict = {}


def _player_index(df: pd.DataFrame) -> dict:
    """Cached lookup over a season's players: full names + first/last tokens, each
    pointing at the highest-games (gsis, display) so name collisions favor the star."""
    # No stats loaded (e.g. offseason: this season's data doesn't exist yet) — return
    # an empty index so player lookups miss cleanly and snapshot-only questions
    # (scoring rules, projections) still answer.
    if df is None or len(df) == 0 or {"player_id", "player_display_name"} - set(df.columns):
        return {"full": {}, "firsts": {}, "lasts": {}}
    season = int(df["season"].iloc[0]) if "season" in df.columns and len(df) else id(df)
    if season in _INDEX_CACHE:
        return _INDEX_CACHE[season]
    full: dict[str, tuple] = {}
    firsts: dict[str, tuple] = {}
    lasts: dict[str, tuple] = {}
    for (gid, disp), games in df.groupby(["player_id", "player_display_name"]).size().items():
        toks = norm_name(str(disp)).split()
        if not toks:
            continue
        rec = (int(games), gid, disp)
        for d, k in ((full, " ".join(toks)), (firsts, toks[0]), (lasts, toks[-1])):
            if k not in d or rec[0] > d[k][0]:
                d[k] = rec
    idx = {"full": full, "firsts": firsts, "lasts": lasts}
    _INDEX_CACHE[season] = idx
    return idx


def find_player(df: pd.DataFrame, text: str):
    """(gsis, display) for the best player named anywhere in `text`. Matches real
    player names with EXACT tokens (handles lowercase, ignores question words)."""
    idx = _player_index(df)
    toks = norm_name(text).split()
    n = len(toks)
    # longest consecutive run that is a full player name (e.g. 'bijan robinson')
    for length in range(min(4, n), 1, -1):
        for i in range(n - length + 1):
            cand = " ".join(toks[i:i + length])
            if cand in idx["full"]:
                _, gid, disp = idx["full"][cand]
                return gid, disp
    # else a single distinctive token = a first or last name (skip question words)
    best = None
    for t in toks:
        if t in _STOP or len(t) < 3:
            continue
        for rec in (idx["firsts"].get(t), idx["lasts"].get(t)):
            if rec and (best is None or rec[0] > best[0]):
                best = rec
    return (best[1], best[2]) if best else (None, None)


def resolve_player(df: pd.DataFrame, name: str):
    """(gsis_id, display_name) for a clean player name (crosswalk first, then the
    robust token matcher)."""
    xw = crosswalk()
    gid = xw.resolve(name)
    if gid is not None and (df["player_id"] == gid).any():
        return gid, df.loc[df["player_id"] == gid, "player_display_name"].iloc[0]
    return find_player(df, name)


# ── tools ─────────────────────────────────────────────────────────────────────
def get_player_stat(ctx: ChatContext, player: str, stat: str,
                    from_week: int = 1, to_week: int | None = None,
                    season: int | None = None) -> str:
    df = ctx.weekly(season)
    gid, disp = resolve_player(df, player)
    if not gid:
        return f"No {season or ctx.season} stats found for a player named '{player}'."
    # default to the fantasy regular season (≤18); the caller can pass a higher
    # to_week to include the NFL playoffs (weeks 19-22 in nflverse).
    to_week = to_week or min(int(df["week"].max()), 18)
    sub = df[(df["player_id"] == gid) & (df["week"] >= from_week) & (df["week"] <= to_week)]
    if sub.empty:
        return f"{disp} has no games in weeks {from_week}-{to_week} of {season or ctx.season}."
    key = stat.lower().strip().replace(" ", "_")
    if key in ("fantasy_points", "points", "fantasy") and ctx.engine():
        sub = sub.assign(fpts=ctx.engine().score_dataframe(sub))
        total = float(sub["fpts"].sum())
        byweek = ", ".join(f"wk{int(r.week)}:{r.fpts:.1f}" for r in sub.itertuples())
        return f"{disp}: {total:.1f} league fantasy pts, weeks {from_week}-{to_week} ({season or ctx.season}). [{byweek}]"
    cols = STAT_ALIASES.get(key)
    if cols is None:
        return f"I don't track the stat '{stat}'. Supported: {_SUPPORTED}, fantasy_points."
    present = [c for c in cols if c in sub.columns]
    total = float(sub[present].fillna(0).to_numpy().sum())
    per = sub.assign(val=sub[present].fillna(0).sum(axis=1))
    byweek = ", ".join(f"wk{int(r.week)}:{r.val:g}" for r in per.itertuples() if r.val)
    return (f"{disp}: {total:{('.1f' if 'yard' in key else 'g')}} {key.replace('_',' ')} "
            f"over weeks {from_week}-{to_week} ({season or ctx.season}), {len(sub)} games. [{byweek}]")


def get_player_game_log(ctx: ChatContext, player: str, season: int | None = None) -> str:
    df = ctx.weekly(season)
    gid, disp = resolve_player(df, player)
    if not gid:
        return f"No {season or ctx.season} stats found for '{player}'."
    sub = df[df["player_id"] == gid].sort_values("week")
    lines = []
    for r in sub.itertuples():
        lines.append(f"wk{int(r.week)} vs {getattr(r,'opponent_team','?')}: "
                     f"{int(getattr(r,'receptions',0) or 0)}rec/{getattr(r,'receiving_yards',0) or 0:.0f}yd, "
                     f"{getattr(r,'rushing_yards',0) or 0:.0f}rush, "
                     f"{int((getattr(r,'passing_tds',0) or 0)+(getattr(r,'rushing_tds',0) or 0)+(getattr(r,'receiving_tds',0) or 0))}TD")
    return f"{disp} {season or ctx.season} game log:\n" + "\n".join(lines)


def get_player_absences(ctx: ChatContext, player: str, season: int | None = None) -> str:
    """Weeks a player has no stat line — the proxy for 'when did X get injured'."""
    df = ctx.weekly(season)
    gid, disp = resolve_player(df, player)
    if not gid:
        return f"No {season or ctx.season} stats found for '{player}'."
    played = sorted(int(w) for w in df.loc[df["player_id"] == gid, "week"].unique())
    if not played:
        return f"{disp} has no games in {season or ctx.season}."
    last_lg = min(int(df["week"].max()), 18)  # fantasy regular season
    missed = [w for w in range(played[0], last_lg + 1) if w not in set(played)]
    first_missed = next((w for w in missed), None)
    note = ("" if not missed else
            " (a single isolated missed week may be a bye, not an injury).")
    return (f"{disp} played weeks {played} in {season or ctx.season}. "
            f"Missed: {missed or 'none'}. First absence after starting the season: "
            f"{first_missed if first_missed else 'none'}.{note}")


def get_projection(ctx: ChatContext, player: str) -> str:
    bi = ctx.board_index or {}
    e = bi.get(norm_name(player))
    if e is None:  # fuzzy
        e = next((v for k, v in bi.items() if norm_name(player) in k), None)
    if not e:
        return (f"No current-week projection on the board for '{player}'. "
                f"(Projections come from the built dashboard for this league.)")
    return (f"{e['name']} ({e.get('pos','?')}) is projected {e['proj']} pts this week "
            f"(model+ESPN+Sleeper consensus), VOR {e['vor']}.")


def get_league_settings(ctx: ChatContext) -> str:
    if not ctx.scoring and not ctx.league_summary:
        return "No league is loaded — open a built league to ask about its settings."
    sc = ", ".join(f"{k}={v}" for k, v in sorted(ctx.scoring.items()) if v) or "n/a"
    te = f" TE-premium: {ctx.te_premium}." if ctx.te_premium else ""
    return f"{ctx.league_summary}\nScoring: {sc}.{te}\nStarting slots: {ctx.roster}."


# ── dispatch + schemas ────────────────────────────────────────────────────────
def run_tool(name: str, inp: dict, ctx: ChatContext) -> str:
    try:
        if name == "get_player_stat":
            return get_player_stat(ctx, inp["player"], inp["stat"],
                                   int(inp.get("from_week", 1)),
                                   int(inp["to_week"]) if inp.get("to_week") else None,
                                   int(inp["season"]) if inp.get("season") else None)
        if name == "get_player_game_log":
            return get_player_game_log(ctx, inp["player"],
                                       int(inp["season"]) if inp.get("season") else None)
        if name == "get_player_absences":
            return get_player_absences(ctx, inp["player"],
                                       int(inp["season"]) if inp.get("season") else None)
        if name == "get_projection":
            return get_projection(ctx, inp["player"])
        if name == "get_league_settings":
            return get_league_settings(ctx)
    except Exception as e:  # noqa: BLE001
        log.warning("tool %s failed: %s", name, e)
        return f"Tool error: {e}"
    return f"Unknown tool {name}."


TOOLS = [
    {"name": "get_player_stat",
     "description": "Total a player's stat over a week range (real nflverse data). "
                    "Use for 'how many TDs/receptions/yards has X had since week N'. "
                    "stat is one of: touchdowns, passing_tds, rushing_tds, receiving_tds, "
                    "receptions, targets, rushing_yards, receiving_yards, passing_yards, "
                    "carries, completions, interceptions, fantasy_points (league-scored).",
     "input_schema": {"type": "object", "properties": {
         "player": {"type": "string"}, "stat": {"type": "string"},
         "from_week": {"type": "integer", "description": "first week, inclusive (default 1)"},
         "to_week": {"type": "integer", "description": "last week, inclusive (default latest)"},
         "season": {"type": "integer"}},
         "required": ["player", "stat"]}},
    {"name": "get_player_absences",
     "description": "Weeks a player has no game (injury/bye proxy). Call this FIRST to find "
                    "the week an event like an injury happened, then aggregate from that week.",
     "input_schema": {"type": "object", "properties": {
         "player": {"type": "string"}, "season": {"type": "integer"}},
         "required": ["player"]}},
    {"name": "get_player_game_log",
     "description": "A player's week-by-week stat line for a season.",
     "input_schema": {"type": "object", "properties": {
         "player": {"type": "string"}, "season": {"type": "integer"}},
         "required": ["player"]}},
    {"name": "get_projection",
     "description": "A player's projected fantasy points for the current week (consensus "
                    "of our model + ESPN + Sleeper). Use for 'how many points is X projected'.",
     "input_schema": {"type": "object", "properties": {"player": {"type": "string"}},
                      "required": ["player"]}},
    {"name": "get_league_settings",
     "description": "This league's scoring rules, roster slots, and format. Use for "
                    "questions about how the user's league scores or is configured.",
     "input_schema": {"type": "object", "properties": {}}},
]
