"""Chatbot data tools + keyless fallback parser (grounded on real nflverse data)."""

from __future__ import annotations

import pytest

from fantasy.chat.agent import _detect_stat, _fallback, _window
from fantasy.chat.tools import (ChatContext, find_player, get_league_settings,
                                get_player_absences, get_player_stat, resolve_player)

SEASON = 2024


@pytest.fixture(scope="module")
def ctx():
    return ChatContext(season=SEASON,
                       scoring={"receptions": 1.0, "passing_tds": 4.0, "rushing_tds": 6.0})


def test_resolve_player_finds_known_star(ctx):
    df = ctx.weekly()
    gid, disp = resolve_player(df, "Ja'Marr Chase")
    assert gid and "Chase" in disp


def test_stat_total_matches_reality(ctx):
    # Chase led the NFL in receptions in 2024 (~127); assert a tight, stable range.
    out = get_player_stat(ctx, "Ja'Marr Chase", "receptions")
    n = int(out.split(":")[1].split("receptions")[0])
    assert 115 <= n <= 135
    assert "2024" in out


def test_touchdowns_since_week_aggregates(ctx):
    full = get_player_stat(ctx, "Ja'Marr Chase", "touchdowns", from_week=1)
    since = get_player_stat(ctx, "Ja'Marr Chase", "touchdowns", from_week=10)
    f = int(full.split(":")[1].split("touchdowns")[0])
    s = int(since.split(":")[1].split("touchdowns")[0])
    assert f >= s >= 0  # a later start can only have <= total


def test_absences_flag_injury_weeks(ctx):
    # McCaffrey missed most of 2024 with injury — there must be missed weeks.
    out = get_player_absences(ctx, "Christian McCaffrey")
    assert "Missed:" in out and "none" not in out.split("Missed:")[1][:8]


def test_unknown_player_is_graceful(ctx):
    out = get_player_stat(ctx, "Nota Realplayer", "receptions")
    assert "No 2024 stats" in out


def test_detect_stat():
    assert _detect_stat("how many receptions") == "receptions"
    assert _detect_stat("rushing yards this year") == "rushing_yards"
    assert _detect_stat("how many touchdowns") in ("touchdowns", "tds", "td")


def test_fallback_parses_player_stat_and_week(ctx):
    out = _fallback("How many receptions has Ja'Marr Chase had since week 5?", ctx)
    assert "get_player_stat" in out["tools_used"]
    assert "Chase" in out["answer"] and "weeks 5-" in out["answer"]


def test_fallback_compound_subject_is_correct(ctx):
    # Subject is Jordan Mason; McCaffrey is only the event reference ("since X injured").
    out = _fallback("How many receptions has Jordan Mason made since "
                    "Christian McCaffrey got injured?", ctx)
    assert "Jordan Mason" in out["answer"]
    assert "McCaffrey" not in out["answer"]
    assert "get_player_absences" in out["tools_used"]  # found the injury week


def test_find_player_lowercase_and_no_false_match(ctx):
    df = ctx.weekly()
    # lowercase full name resolves
    gid, disp = find_player(df, "how many tds did bijan robinson score")
    assert disp == "Bijan Robinson"
    # lowercase single first name resolves to the star
    gid2, disp2 = find_player(df, "how many td is bijan score this year")
    assert disp2 == "Bijan Robinson"
    # a question word must NOT match a player (the old 'How' -> 'Howden' bug)
    gid3, _ = find_player(df, "How many points")
    assert gid3 is None


def test_fallback_single_week(ctx):
    out = _fallback("How many td did bijan robinson score in week 4", ctx)
    assert "Bijan Robinson" in out["answer"] and "weeks 4-4" in out["answer"]


def test_window_detection(ctx):
    assert _window("tds since week 3", "", ctx, []) == (3, None)
    assert _window("yards in week 7", "", ctx, []) == (7, 7)
    assert _window("rushing yards weeks 1 to 5", "", ctx, []) == (1, 5)


def test_fallback_league_settings():
    c = ChatContext(season=SEASON, league_summary="My League (2024)",
                    scoring={"receptions": 1.0, "passing_tds": 4.0})
    out = get_league_settings(c)
    assert "receptions=1.0" in out and "passing_tds=4.0" in out
