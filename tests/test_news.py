"""News layer: deterministic classification + alert generation (no network)."""

from __future__ import annotations

from fantasy.league_state import LeagueSnapshot
from fantasy.news.espn_news import _classify
from fantasy.news.ingest import make_alerts
from fantasy.news.models import EventType, PlayerSignal


def test_injury_keyword_classification():
    assert _classify("Star RB ruled out for Sunday") == EventType.injury_out
    assert _classify("WR listed as doubtful") == EventType.injury_doubtful
    assert _classify("QB questionable with ankle") == EventType.injury_questionable
    assert _classify("TE placed on injured reserve") == EventType.ir
    assert _classify("RB returns to practice, will play") == EventType.injury_return
    assert _classify("Team signs free agent lineman") == EventType.news


def _snap():
    return LeagueSnapshot(
        season=2024, week=8, my_team_id=1,
        teams={1: ["mine1", "mine2"], 2: ["opp1"]}, free_agents=["fa1", "fa2"],
        names={"mine1": "My Star", "fa1": "Hot Pickup"},
        positions={"mine1": "RB", "fa1": "WR"},
    )


def test_alerts_only_for_my_roster_and_trending_targets():
    snap = _snap()
    signals = [
        PlayerSignal(player_id="mine1", player_name="My Star", event_type=EventType.injury_out,
                     summary="ruled out", source="espn"),
        PlayerSignal(player_id="fa1", player_name="Hot Pickup", event_type=EventType.trending_add,
                     summary="9000 adds", source="sleeper"),
        PlayerSignal(player_id="opp1", player_name="Their Guy", event_type=EventType.injury_out,
                     summary="ruled out", source="espn"),   # opponent — ignored
        PlayerSignal(player_id="zzz", player_name="Random", event_type=EventType.news,
                     summary="contract talk", source="espn"),  # not mine — ignored
    ]
    alerts = make_alerts(snap, signals)
    titles = " ".join(a.title for a in alerts)
    assert "My Star" in titles            # my injured player
    assert "Hot Pickup" in titles         # trending FA available
    assert "Their Guy" not in titles      # opponent injury not alerted
    assert "Random" not in titles
    assert all(a.kind.value == "alert" for a in alerts)


def test_alert_includes_beneficiaries():
    snap = _snap()
    sig = PlayerSignal(player_id="mine1", player_name="My Star", event_type=EventType.injury_out,
                       summary="out", source="espn", beneficiaries=["Backup RB"])
    [alert] = make_alerts(snap, [sig])
    assert "Backup RB" in alert.detail
