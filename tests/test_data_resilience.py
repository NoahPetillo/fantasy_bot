"""Data-layer resilience to not-yet-published seasons.

Projecting the upcoming season (e.g. 2026 in the preseason) legitimately asks
nflverse for data that doesn't exist yet. nflreadpy raises on a future season
("Season must be between 2012 and 2025"); the loaders must skip it and degrade,
never crash the build.
"""

from __future__ import annotations

import pandas as pd

import fantasy.data.nfl as nfl_mod


class _FakeFrame:
    """Mimics the polars frame nflreadpy returns (.to_pandas())."""

    def __init__(self, df: pd.DataFrame):
        self._df = df

    def to_pandas(self) -> pd.DataFrame:
        return self._df


def test_player_snap_share_skips_unpublished_season(tmp_path, monkeypatch):
    monkeypatch.setattr(nfl_mod.settings, "data_dir", tmp_path)

    def fake_load_snap_counts(seasons):
        if any(int(s) >= 2026 for s in seasons):
            raise ValueError("Season must be between 2012 and 2025")
        return _FakeFrame(pd.DataFrame(
            {"pfr_player_id": ["A"], "season": [2025], "week": [1], "offense_pct": [0.8]}))

    monkeypatch.setattr("nflreadpy.load_snap_counts", fake_load_snap_counts)
    monkeypatch.setattr(nfl_mod, "load_player_ids",
                        lambda: pd.DataFrame({"pfr_id": ["A"], "gsis_id": ["00-1"]}))

    # The mixed request (published 2025 + unpublished 2026) must not raise.
    out = nfl_mod.player_snap_share([2025, 2026])
    assert list(out["season"].unique()) == [2025]  # 2026 skipped, 2025 kept
    assert out.iloc[0]["player_id"] == "00-1"
    # Partial result must NOT be cached (would poison the key once 2026 publishes).
    assert not (tmp_path / "cache" / "snaps_2025_2026.parquet").exists()


def test_player_snap_share_all_unpublished_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(nfl_mod.settings, "data_dir", tmp_path)

    def fake_load_snap_counts(seasons):
        raise ValueError("Season must be between 2012 and 2025")

    monkeypatch.setattr("nflreadpy.load_snap_counts", fake_load_snap_counts)
    monkeypatch.setattr(nfl_mod, "load_player_ids",
                        lambda: pd.DataFrame({"pfr_id": ["A"], "gsis_id": ["00-1"]}))

    out = nfl_mod.player_snap_share([2026])  # nothing available -> empty, no crash
    assert out.empty
    assert list(out.columns) == ["season", "week", "player_id", "offense_pct"]
