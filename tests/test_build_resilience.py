"""The full build must not 500 when a requested season isn't published yet.

Reproduces the preseason case: it's before the current season's first games, so
nflverse has no ``stats_player_week_<year>.parquet`` and the download 404s. The
data spine skips the missing season, and ``assemble`` degrades to the shell view
instead of fabricating a lineup/waivers/trades from an empty value board.
"""

from __future__ import annotations

import pandas as pd

import fantasy.api.dashboard_data as dd
import fantasy.data.nfl as nfl_mod


# ── data spine: skip a season that isn't published yet ────────────────────────
def test_load_weekly_skips_unpublished_season(monkeypatch):
    import nflreadpy

    class _Wrap:
        def __init__(self, df): self._df = df
        def to_pandas(self): return self._df

    def fake_load(seasons):
        # Any batch containing the unpublished season 404s (like the real client);
        # a single published season loads fine.
        if 2026 in seasons:
            raise ConnectionError("404 Not Found: stats_player_week_2026.parquet")
        return _Wrap(pd.DataFrame({"season": list(seasons),
                                   "player_id": [f"p{s}" for s in seasons],
                                   "position": ["RB"] * len(seasons)}))

    monkeypatch.setattr(nflreadpy, "load_player_stats", fake_load)
    df = nfl_mod.load_weekly([2024, 2026], refresh=True)  # 2026 not out yet
    assert not df.empty and set(df["season"]) == {2024}   # skipped 2026, no crash


def test_load_weekly_raises_only_when_nothing_loads(monkeypatch):
    import nflreadpy

    def all_missing(seasons):
        raise ConnectionError("404 for every season")

    monkeypatch.setattr(nflreadpy, "load_player_stats", all_missing)
    try:
        nfl_mod.load_weekly([2026], refresh=True)
        raised = False
    except ConnectionError:
        raised = True
    assert raised  # a genuine total outage still surfaces


# ── assemble: empty board → shell view (preseason) ────────────────────────────
class _Scoring:
    value = "PPR"


class _League:
    league_id = 998877
    team_count = 12
    scoring_format = _Scoring()

    def summary(self): return "12-team PPR"


class _Team:
    def __init__(self, tid, name):
        self.team_id, self.team_name = tid, name
        self.wins = self.losses = 0
        self.points_for = 0.0


class _Client:
    def week_projections(self, wk): return {}
    def teams(self): return [_Team(1, "Alpha"), _Team(2, "Bravo")]


class _Service:
    def project(self, season, week, **kw): return pd.DataFrame()  # no data yet


def test_assemble_empty_board_returns_shell(monkeypatch):
    monkeypatch.setattr("fantasy.orchestrator.cycle.fetch_expert_signals", lambda: [])
    out = dd.assemble(_Service(), _League(), store=None, season=2026, week=1,
                      client=_Client(), my_team_id=1)
    assert out["team"]["shell"] is True          # degraded to the preseason shell
    assert out["report"] is None                 # no report card fabricated
    assert len(out["standings"]) == 2            # real league data still shown
    assert out["waivers"] == [] and out["trades"] == []
