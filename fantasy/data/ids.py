"""Central ID crosswalk — the linchpin join across data sources.

Maps Sleeper / ESPN / PFR ids to the canonical gsis ``player_id`` (the key used
by nflverse weekly stats and our projection board), plus name/position lookups.
Built once from ``load_ff_playerids`` and reused by the news layer, the live ESPN
snapshot, and the snap-count loader.
"""

from __future__ import annotations

import functools
import re

from fantasy.data.nfl import load_player_ids

_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def norm_name(name: str) -> str:
    """Normalize a player name for cross-source matching."""
    n = (name or "").lower().replace(".", "").replace("'", "").replace("-", " ")
    n = _SUFFIX.sub("", n)
    return " ".join(n.split())


class Crosswalk:
    def __init__(self):
        df = load_player_ids()
        cols = {c.lower(): c for c in df.columns}
        g = cols.get("gsis_id")
        self.sleeper_to_gsis: dict[str, str] = {}
        self.espn_to_gsis: dict[str, str] = {}
        self.gsis_to_name: dict[str, str] = {}
        self.gsis_to_pos: dict[str, str] = {}
        name_c = cols.get("name") or cols.get("merge_name")
        pos_c = cols.get("position")
        for _, r in df.iterrows():
            gid = r.get(g)
            if not isinstance(gid, str) or not gid:
                continue
            sl, es = cols.get("sleeper_id"), cols.get("espn_id")
            if sl and r.get(sl) and str(r[sl]).strip():
                self.sleeper_to_gsis[str(r[sl]).split(".")[0]] = gid
            if es and r.get(es) and str(r[es]).strip():
                self.espn_to_gsis[str(r[es]).split(".")[0]] = gid
            if name_c:
                self.gsis_to_name[gid] = r.get(name_c)
            if pos_c:
                self.gsis_to_pos[gid] = r.get(pos_c)

        # Name+position index for sources that key on names (e.g. FFC ADP).
        self.name_pos_to_gsis: dict[tuple[str, str], str] = {}
        for gid, nm in self.gsis_to_name.items():
            pos = self.gsis_to_pos.get(gid)
            if isinstance(nm, str) and pos:
                self.name_pos_to_gsis[(norm_name(nm), pos)] = gid

    def resolve(self, name: str, position: str | None = None) -> str | None:
        """Best-effort gsis id from a display name (+ optional position)."""
        key = norm_name(name)
        if position:
            gid = self.name_pos_to_gsis.get((key, position))
            if gid:
                return gid
        # fall back to any position match on the name
        for (n, _p), gid in self.name_pos_to_gsis.items():
            if n == key:
                return gid
        return None

    def from_sleeper(self, sleeper_id: str) -> str | None:
        return self.sleeper_to_gsis.get(str(sleeper_id).split(".")[0])

    def from_espn(self, espn_id) -> str | None:
        return self.espn_to_gsis.get(str(espn_id).split(".")[0])

    def name(self, gsis_id: str) -> str:
        return self.gsis_to_name.get(gsis_id, gsis_id)


@functools.lru_cache(maxsize=1)
def crosswalk() -> Crosswalk:
    return Crosswalk()
