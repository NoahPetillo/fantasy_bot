"""ESPN-projection overlay: ESPN is primary, our model fills gaps."""

from __future__ import annotations

import pandas as pd

from fantasy.projections.service import overlay_espn


def _cur():
    return pd.DataFrame({
        "player_id": ["a", "b", "c"],
        "position": ["RB", "WR", "TE"],
        "proj": [10.0, 12.0, 8.0],  # our model
    })


def test_overlay_uses_espn_where_available():
    cur = overlay_espn(_cur(), {"a": 15.0, "c": 9.5})  # b missing
    by = dict(zip(cur["player_id"], cur["proj"]))
    src = dict(zip(cur["player_id"], cur["proj_source"]))
    assert by["a"] == 15.0 and src["a"] == "espn"   # ESPN overrides
    assert by["b"] == 12.0 and src["b"] == "model"  # gap -> our model
    assert by["c"] == 9.5 and src["c"] == "espn"


def test_overlay_none_keeps_model():
    cur = overlay_espn(_cur(), None)
    assert (cur["proj_source"] == "model").all()
    assert list(cur["proj"]) == [10.0, 12.0, 8.0]
