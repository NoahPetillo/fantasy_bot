"""League content engine: detection, ranking, captions, proposal/dedup.

All deterministic — fake box-score objects duck-type espn-api's BoxScore/BoxPlayer,
so nothing here hits the network, an LLM, or a renderer.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import fantasy.moments.publisher as publisher
from fantasy.moments import content, cycle, roasts
from fantasy.moments.activity import detect_trades, detect_waivers
from fantasy.moments.content import write_caption
from fantasy.moments.cycle import activity_cycle, content_cycle
from fantasy.moments.detector import detect_moments
from fantasy.moments.models import MATCHUP_TYPES, Moment, MomentType
from fantasy.moments.score import rank_and_select
from fantasy.moments.standings import detect_rivalries, detect_streaks
from fantasy.orchestrator.models import ProposalKind
from fantasy.orchestrator.store import Store


def test_content_engine_uses_decoupled_config():
    """Every moments module reads the self-contained content config, NOT the app's
    global settings — so the multi-tenant refactor of fantasy/config.py can't break it."""
    from fantasy.moments.config import content_config
    assert content.settings is content_config
    assert cycle.settings is content_config
    assert roasts.settings is content_config


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    # Never let real .env LLM keys leak into tests (no live API calls).
    monkeypatch.setattr(content.settings, "anthropic_api_key", None)
    monkeypatch.setattr(content.settings, "xai_api_key", None)
    monkeypatch.setattr(content.settings, "groq_api_key", None)
    roasts.reset_cache()
    yield
    roasts.reset_cache()


def _player(name, pts, slot, proj=0.0, eligible=None, pid=None):
    return SimpleNamespace(name=name, points=pts, projected_points=proj, slot_position=slot,
                           eligibleSlots=eligible or [slot], playerId=pid or name)


def _box(hn, hid, hs, an, aid, asc, home_lineup=None, away_lineup=None):
    home = SimpleNamespace(team_id=hid, team_name=hn, team_abbrev=hn[:4])
    away = SimpleNamespace(team_id=aid, team_name=an, team_abbrev=an[:4])
    return SimpleNamespace(home_team=home, home_score=hs, home_lineup=home_lineup or [],
                           away_team=away, away_score=asc, away_lineup=away_lineup or [])


def _league_boxes():
    # scores [130,128,140,70,100,95] -> median 114
    return [
        _box("Alpha", 1, 130.0, "Bravo", 2, 128.0),     # nail-biter (margin 2)
        _box("Charlie", 3, 140.0, "Delta", 4, 70.0),    # blowout (margin 70); high=140, low=70
        _box("Echo", 5, 100.0, "Foxtrot", 6, 95.0),     # Echo lucky (won below median)
    ]


def _types(moments):
    return {m.type for m in moments}


# ── detection ────────────────────────────────────────────────────────────────
def test_detects_core_moment_types():
    moments = detect_moments(_league_boxes(), 2025, 7)
    t = _types(moments)
    assert MomentType.nailbiter in t
    assert MomentType.blowout in t
    assert MomentType.high_score in t
    assert MomentType.low_score in t
    assert MomentType.unlucky in t   # Bravo: 128 > median, still lost
    assert MomentType.lucky in t     # Echo: 100 < median, still won


def test_nailbiter_targets_loser_and_orders_winner_first():
    moments = detect_moments(_league_boxes(), 2025, 7)
    nb = next(m for m in moments if m.type == MomentType.nailbiter)
    assert nb.team_id == 2                       # loser (Bravo) is the subject
    assert nb.score_a > nb.score_b               # winner shown first
    assert nb.team_a == "Alpha" and nb.team_b == "Bravo"
    assert "2.0" in nb.big_stat                  # margin


def test_blowout_threshold_not_triggered_for_moderate_margin():
    # 30-point win is neither a nail-biter (<5) nor a blowout (>40).
    moments = detect_moments([_box("X", 1, 120.0, "Y", 2, 90.0)], 2025, 1)
    assert MomentType.nailbiter not in _types(moments)
    assert MomentType.blowout not in _types(moments)


def test_high_low_scores_pick_extremes():
    moments = detect_moments(_league_boxes(), 2025, 7)
    hi = next(m for m in moments if m.type == MomentType.high_score)
    lo = next(m for m in moments if m.type == MomentType.low_score)
    assert hi.team_a == "Charlie" and "140" in hi.big_stat
    assert lo.team_a == "Delta" and "70" in lo.big_stat


def test_bench_blunder_finds_biggest_eligible_swap():
    starter = _player("Started RB", 3.0, "RB", eligible=["RB", "RB/WR/TE"])
    benched = _player("Bench Stud", 24.0, "BE", eligible=["RB", "RB/WR/TE"])
    box = _box("Boom", 1, 110.0, "Bust", 2, 100.0,
               home_lineup=[starter, benched])
    moments = detect_moments([box], 2025, 3)
    bb = next(m for m in moments if m.type == MomentType.bench_blunder)
    assert bb.player == "Bench Stud"
    assert bb.team_id == 1
    assert "21.0" in bb.big_stat   # 24.0 - 3.0


def test_bench_blunder_requires_slot_eligibility():
    # A bench QB can't have replaced a started RB -> no blunder from this pair.
    starter = _player("RB1", 4.0, "RB", eligible=["RB", "RB/WR/TE"])
    bench_qb = _player("QB2", 30.0, "BE", eligible=["QB"])
    box = _box("A", 1, 100.0, "B", 2, 90.0, home_lineup=[starter, bench_qb])
    assert MomentType.bench_blunder not in _types(detect_moments([box], 2025, 3))


def test_boom_and_bust_vs_projection():
    boomer = _player("Boomer", 32.0, "WR", proj=10.0)
    buster = _player("Buster", 2.0, "RB", proj=19.0)
    box = _box("A", 1, 120.0, "B", 2, 119.0, home_lineup=[boomer, buster])
    t = _types(detect_moments([box], 2025, 4))
    assert MomentType.boom in t
    assert MomentType.bust in t


def test_empty_or_unplayed_week_yields_nothing():
    assert detect_moments([], 2025, 1) == []
    assert detect_moments([_box("A", 1, 0.0, "B", 2, 0.0)], 2025, 1) == []


def test_bye_side_counts_for_superlatives_not_matchups():
    bye = SimpleNamespace(home_team=SimpleNamespace(team_id=9, team_name="Lonely", team_abbrev="LONE"),
                          home_score=200.0, home_lineup=[],
                          away_team=None, away_score=0.0, away_lineup=[])
    moments = detect_moments([*_league_boxes(), bye], 2025, 7)
    hi = next(m for m in moments if m.type == MomentType.high_score)
    assert hi.team_a == "Lonely"                       # counts toward high score
    assert MomentType.nailbiter in _types(moments)      # real matchups still detected


# ── ranking / selection ──────────────────────────────────────────────────────
def test_rank_and_select_takes_top_n():
    moments = detect_moments(_league_boxes(), 2025, 7)
    top = rank_and_select(moments, n=2)
    assert len(top) == 2
    assert top[0].spice >= top[1].spice
    assert top == sorted(top, key=lambda m: m.spice, reverse=True)


def test_per_matchup_cap_limits_angles_on_same_game():
    # Two moments on the same matchup key; cap of 1 keeps only the spicier.
    a = Moment(type=MomentType.nailbiter, season=2025, week=1, headline="h", blurb="b",
               spice=90, dedup_key="1-2")
    b = Moment(type=MomentType.lucky, season=2025, week=1, headline="h", blurb="b",
               spice=80, dedup_key="1-2")
    c = Moment(type=MomentType.blowout, season=2025, week=1, headline="h", blurb="b",
               spice=70, dedup_key="3-4")
    chosen = rank_and_select([a, b, c], n=3, max_per_matchup=1)
    assert a in chosen and c in chosen and b not in chosen


# ── caption fallback (no LLM) ──────────────────────────────────────────────────
def _no_llm(monkeypatch):
    monkeypatch.setattr(content.settings, "anthropic_api_key", None)
    monkeypatch.setattr(content.settings, "xai_api_key", None)


def test_caption_fallback_without_key(monkeypatch):
    _no_llm(monkeypatch)
    monkeypatch.setattr(content.settings, "content_voice", "group_chat")
    m = Moment(type=MomentType.nailbiter, season=2025, week=7,
               headline="Alpha survives Bravo by 2.0",
               blurb="Alpha beat Bravo by just 2.0 points.", spice=88)
    cap = write_caption(m)
    assert "Alpha beat Bravo" in cap
    assert "#" not in cap                       # group-chat voice = no hashtags


def test_caption_fallback_instagram_adds_hashtags(monkeypatch):
    _no_llm(monkeypatch)
    monkeypatch.setattr(content.settings, "content_voice", "instagram")
    m = Moment(type=MomentType.blowout, season=2025, week=7, headline="h",
               blurb="Charlie blew out Delta.", spice=88)
    cap = write_caption(m)
    assert "#" in cap and "fuck" not in cap.lower()   # public-safe


def test_savage_fallback_roasts_by_name(monkeypatch):
    _no_llm(monkeypatch)
    monkeypatch.setattr(content.settings, "content_voice", "group_chat")
    m = Moment(type=MomentType.bench_blunder, season=2025, week=7,
               headline="h", blurb="Nick benched a stud.", spice=90, team_id=1, manager="Nick")
    cap = write_caption(m).lower()
    assert "nick" in cap            # called out by name
    assert "fuck" in cap            # savage voice has teeth
    assert "#" not in cap           # group chat = no hashtags


# ── caption LLM provider (Groq / xAI-Grok / Anthropic) ─────────────────────────
def test_provider_auto_prefers_groq(monkeypatch):
    monkeypatch.setattr(content.settings, "content_llm_provider", "auto")
    monkeypatch.setattr(content.settings, "groq_api_key", "gsk_key")
    monkeypatch.setattr(content.settings, "xai_api_key", "xai-key")
    monkeypatch.setattr(content.settings, "anthropic_api_key", "anth-key")
    monkeypatch.setattr(content.settings, "content_llm_model", None)
    monkeypatch.setattr(content.settings, "groq_model", "llama-3.3-70b-versatile")
    prov, model, key = content._resolve_provider()
    assert prov == "groq" and key == "gsk_key" and model == "llama-3.3-70b-versatile"


def test_provider_auto_chain_xai_then_anthropic(monkeypatch):
    # No groq key -> xAI wins over Anthropic.
    monkeypatch.setattr(content.settings, "content_llm_provider", "auto")
    monkeypatch.setattr(content.settings, "xai_api_key", "xai-key")
    monkeypatch.setattr(content.settings, "anthropic_api_key", "anth-key")
    prov, model, _ = content._resolve_provider()
    assert prov == "xai" and "grok" in model.lower()


def test_explicit_provider_without_key_yields_none(monkeypatch):
    monkeypatch.setattr(content.settings, "content_llm_provider", "anthropic")
    monkeypatch.setattr(content.settings, "groq_api_key", "gsk_key")   # present but not selected
    assert content._resolve_provider() == (None, None, None)


def test_model_override_respected(monkeypatch):
    monkeypatch.setattr(content.settings, "content_llm_provider", "groq")
    monkeypatch.setattr(content.settings, "groq_api_key", "gsk_key")
    monkeypatch.setattr(content.settings, "content_llm_model", "openai/gpt-oss-120b")
    prov, model, _ = content._resolve_provider()
    assert prov == "groq" and model == "openai/gpt-oss-120b"


def test_write_caption_routes_to_groq(monkeypatch):
    monkeypatch.setattr(content.settings, "content_llm_provider", "auto")
    monkeypatch.setattr(content.settings, "groq_api_key", "gsk_key")
    monkeypatch.setattr(content, "_groq_complete", lambda prompt, model, key: "GROQ SAYS YOU SUCK")
    m = Moment(type=MomentType.blowout, season=2025, week=1, headline="h", blurb="b", spice=80)
    assert write_caption(m) == "GROQ SAYS YOU SUCK"


def test_manager_enrichment_from_owners(tmp_path):
    from fantasy.moments.cycle import _enrich_managers
    A = SimpleNamespace(team_id=1, team_name="A", team_abbrev="A",
                        owners=[{"firstName": "Nick", "lastName": "V"}])
    m = Moment(type=MomentType.low_score, season=2025, week=3, headline="h", blurb="b",
               spice=40, team_id=1)
    _enrich_managers([m], [A])
    assert m.manager == "Nick"


# ── roast book (per-league inside jokes, occasional) ───────────────────────────
def _roast_file(tmp_path, league_id="999"):
    p = tmp_path / "roasts.yaml"
    p.write_text(
        f'leagues:\n  "{league_id}":\n    managers:\n'
        '      - name: Cam\n        notes:\n          - "skipped the draft to hang with friends"\n'
        '      - name: James\n        nickname: Stone\n        notes:\n'
        '          - "brags about old championships"\n'
    )
    return p


def _point_at(monkeypatch, path, league_id=999, freq=1):
    monkeypatch.setattr(roasts.settings, "content_roasts_file", path)
    monkeypatch.setattr(roasts.settings, "espn_league_id", league_id)
    monkeypatch.setattr(roasts.settings, "content_roast_frequency", freq)
    roasts.reset_cache()


def test_roast_fires_for_known_manager(monkeypatch, tmp_path):
    _point_at(monkeypatch, _roast_file(tmp_path), freq=1)
    assert "skipped the draft" in (roasts.roast_for("Cam", 5) or "")


def test_roast_uses_nickname(monkeypatch, tmp_path):
    _point_at(monkeypatch, _roast_file(tmp_path), freq=1)
    out = roasts.roast_for("James", 3) or ""
    assert "Stone" in out and "championships" in out


def test_roast_is_occasional_not_every_week(monkeypatch, tmp_path):
    _point_at(monkeypatch, _roast_file(tmp_path), freq=3)
    fires = [w for w in range(1, 31) if roasts.roast_for("Cam", w)]
    assert 0 < len(fires) < 30          # shows up sometimes, not every week


def test_roast_none_for_unknown_manager_or_league(monkeypatch, tmp_path):
    p = _roast_file(tmp_path)
    _point_at(monkeypatch, p, league_id=999, freq=1)
    assert roasts.roast_for("Nobody", 5) is None
    monkeypatch.setattr(roasts.settings, "espn_league_id", 424242)  # no block
    roasts.reset_cache()
    assert roasts.roast_for("Cam", 5) is None


def test_roast_graceful_without_file(monkeypatch, tmp_path):
    _point_at(monkeypatch, tmp_path / "does_not_exist.yaml", freq=1)
    assert roasts.roast_for("Cam", 5) is None


# ── full cycle: proposals + idempotency ────────────────────────────────────────
class _FakeClient:
    def __init__(self, boxes, teams=None):
        self._boxes = boxes
        self._teams = teams or []

    def box_scores(self, week):
        return self._boxes

    def teams(self):
        return self._teams


def test_content_cycle_builds_moment_proposals_and_dedups(tmp_path):
    store = Store(path=tmp_path / "t.sqlite")
    client = _FakeClient(_league_boxes())
    first = content_cycle(client, 2025, 7, store=store, notifier=None,
                          per_week=3, generate=False)
    assert first, "expected fresh moment proposals"
    assert all(p.kind == ProposalKind.moment for p in first)
    assert all(p.payload["channel"] == "discord" for p in first)
    # Re-running the same week produces no new proposals (idempotency by moment id).
    again = content_cycle(client, 2025, 7, store=store, notifier=None,
                          per_week=3, generate=False)
    assert again == []
    store.close()


# ── Phase 2: streaks ───────────────────────────────────────────────────────────
def _team(tid, name, streak_type="WIN", streak_len=0, schedule=None, outcomes=None, scores=None):
    return SimpleNamespace(team_id=tid, team_name=name, team_abbrev=name[:4],
                           streak_type=streak_type, streak_length=streak_len,
                           schedule=schedule or [], outcomes=outcomes or [], scores=scores or [])


def test_detect_streaks_picks_longest_hot_and_cold():
    teams = [_team(1, "Hot", "WIN", 6), _team(2, "Warm", "WIN", 3),
             _team(3, "Cold", "LOSS", 5), _team(4, "Meh", "WIN", 2)]  # Meh below min
    ms = detect_streaks(teams, 2025, 10, min_len=3)
    by = {m.type: m for m in ms}
    assert by[MomentType.hot_streak].team_id == 1 and "6" in by[MomentType.hot_streak].big_stat
    assert by[MomentType.cold_streak].team_id == 3 and "5" in by[MomentType.cold_streak].big_stat


def test_streak_below_min_not_flagged():
    assert detect_streaks([_team(1, "A", "WIN", 2)], 2025, 5, min_len=3) == []


# ── Phase 2: rivalries ─────────────────────────────────────────────────────────
def test_detect_rivalries_emits_with_series_record():
    # wk3 (idx2): Alpha beats Bravo. They also met wk1 (Alpha lost). Series even 1-1.
    A = _team(1, "Alpha", schedule=[2, 9, 2], outcomes=["L", "W", "W"], scores=[100, 120, 140])
    B = _team(2, "Bravo", schedule=[1, 8, 1], outcomes=["W", "L", "L"], scores=[110, 90, 130])
    other = _team(9, "Other", schedule=[8, 1, 8], outcomes=["W", "L", "W"], scores=[99, 99, 99])
    ms = detect_rivalries([A, B, other], 2025, 3, pairs=[["Alpha", "Bravo"]])
    assert len(ms) == 1
    m = ms[0]
    assert m.type == MomentType.rivalry and m.type in MATCHUP_TYPES
    assert m.team_a == "Alpha" and m.team_b == "Bravo"     # Alpha won this week
    assert m.score_a == 140 and m.score_b == 130
    assert m.big_stat == "1-1"


def test_rivalry_resolves_tokens_by_id():
    A = _team(5, "Alpha", schedule=[6], outcomes=["W"], scores=[100])
    B = _team(6, "Bravo", schedule=[5], outcomes=["L"], scores=[90])
    ms = detect_rivalries([A, B], 2025, 1, pairs=[["5", "6"]])
    assert len(ms) == 1 and ms[0].team_a == "Alpha"


def test_no_rivalry_when_pair_didnt_play_this_week():
    A = _team(1, "Alpha", schedule=[9, 9, 9], outcomes=["W", "W", "W"])
    B = _team(2, "Bravo", schedule=[8, 8, 8], outcomes=["L", "L", "L"])
    assert detect_rivalries([A, B], 2025, 3, pairs=[["Alpha", "Bravo"]]) == []


def test_no_rivalries_without_config():
    A = _team(1, "Alpha", schedule=[2], outcomes=["W"], scores=[100])
    B = _team(2, "Bravo", schedule=[1], outcomes=["L"], scores=[90])
    assert detect_rivalries([A, B], 2025, 1, pairs=None) == []


# ── Phase 2: trades + waivers (activity feed) ──────────────────────────────────
def _activity(date_ms, actions):
    return SimpleNamespace(date=date_ms, actions=actions)


def _pl(name):
    return SimpleNamespace(name=name)


def test_detect_trades_groups_received_players():
    tA = SimpleNamespace(team_id=1, team_name="Alpha", team_abbrev="ALPH")
    tB = SimpleNamespace(team_id=2, team_name="Bravo", team_abbrev="BRAV")
    acts = [_activity(1700000000000, [
        (tA, "TRADE_SENT", _pl("Star RB"), 0), (tB, "TRADE_RECEIVED", _pl("Star RB"), 0),
        (tB, "TRADE_SENT", _pl("WR1"), 0), (tA, "TRADE_RECEIVED", _pl("WR1"), 0),
        (tB, "TRADE_SENT", _pl("WR2"), 0), (tA, "TRADE_RECEIVED", _pl("WR2"), 0),
    ])]
    [m] = detect_trades(acts, 2025)
    assert m.type == MomentType.trade and m.week == 0 and m.period_label
    assert "Alpha gets WR1, WR2" in m.blurb
    assert "Bravo gets Star RB" in m.blurb


def test_detect_waivers_flags_only_big_bids():
    t = SimpleNamespace(team_id=3, team_name="Gamma", team_abbrev="GAM")
    acts = [_activity(1700000000000, [
        (t, "WAIVER ADDED", _pl("Hot FA"), 42),
        (t, "WAIVER ADDED", _pl("Cheap FA"), 1),
        (t, "FA ADDED", _pl("Free Guy"), 0),
    ])]
    [m] = detect_waivers(acts, 2025, min_bid=15)
    assert m.big_stat == "$42" and m.player == "Hot FA" and m.team_id == 3 and m.week == 0


class _ActClient:
    def __init__(self, acts=None, raises=False):
        self._acts, self._raises = acts or [], raises

    def recent_activity(self, size=60):
        if self._raises:
            raise RuntimeError("League 123 does not exist")
        return self._acts


def test_activity_cycle_builds_and_dedups(tmp_path):
    t = SimpleNamespace(team_id=3, team_name="Gamma", team_abbrev="GAM")
    acts = [_activity(1700000000000, [(t, "WAIVER ADDED", _pl("Hot FA"), 42)])]
    store = Store(path=tmp_path / "a.sqlite")
    first = activity_cycle(_ActClient(acts), 2025, store=store, notifier=None, generate=False)
    assert len(first) == 1 and first[0].kind == ProposalKind.moment
    assert activity_cycle(_ActClient(acts), 2025, store=store, notifier=None, generate=False) == []
    store.close()


def test_activity_cycle_graceful_when_feed_unavailable(tmp_path):
    store = Store(path=tmp_path / "b.sqlite")
    assert activity_cycle(_ActClient(raises=True), 2025, store=store, notifier=None,
                          generate=False) == []
    store.close()


# ── auto-post (hands-off deploy mode) ──────────────────────────────────────────
def _stub_generation(monkeypatch):
    monkeypatch.setattr(cycle, "render_card", lambda m, out_dir=None: None)
    monkeypatch.setattr(cycle, "write_caption", lambda m: "savage cap")


def test_autopost_posts_and_marks_executed(monkeypatch, tmp_path):
    _stub_generation(monkeypatch)
    monkeypatch.setattr(cycle.settings, "content_autopost", True)
    monkeypatch.setattr(cycle.settings, "content_autopost_min_spice", 0.0)
    posted: list[str] = []
    monkeypatch.setattr(publisher, "publish_moment", lambda p: posted.append(p.id) or "discord:1")
    store = Store(path=tmp_path / "ap.sqlite")
    fresh = content_cycle(_FakeClient(_league_boxes()), 2025, 7, store=store,
                          notifier=None, per_week=2, generate=True)
    assert fresh and len(posted) == len(fresh)          # every fresh moment posted
    assert all(store.get(p.id).status.value == "executed" for p in fresh)
    store.close()


def test_no_autopost_when_disabled(monkeypatch, tmp_path):
    _stub_generation(monkeypatch)
    monkeypatch.setattr(cycle.settings, "content_autopost", False)
    called: list[int] = []
    monkeypatch.setattr(publisher, "publish_moment", lambda p: called.append(1) or "x")
    store = Store(path=tmp_path / "ap2.sqlite")
    fresh = content_cycle(_FakeClient(_league_boxes()), 2025, 7, store=store,
                          notifier=None, per_week=2, generate=True)
    assert fresh and not called                          # nothing auto-posted
    assert all(store.get(p.id).status.value == "proposed" for p in fresh)
    store.close()


def test_autopost_respects_spice_threshold(monkeypatch, tmp_path):
    _stub_generation(monkeypatch)
    monkeypatch.setattr(cycle.settings, "content_autopost", True)
    monkeypatch.setattr(cycle.settings, "content_autopost_min_spice", 999.0)  # nothing qualifies
    called: list[int] = []
    monkeypatch.setattr(publisher, "publish_moment", lambda p: called.append(1) or "x")
    store = Store(path=tmp_path / "ap3.sqlite")
    fresh = content_cycle(_FakeClient(_league_boxes()), 2025, 7, store=store,
                          notifier=None, per_week=2, generate=True)
    assert fresh and not called                          # threshold too high
    store.close()


def test_autopost_failure_falls_back_to_proposed(monkeypatch, tmp_path):
    _stub_generation(monkeypatch)
    monkeypatch.setattr(cycle.settings, "content_autopost", True)
    monkeypatch.setattr(cycle.settings, "content_autopost_min_spice", 0.0)
    monkeypatch.setattr(publisher, "publish_moment", lambda p: None)   # e.g. no webhook
    store = Store(path=tmp_path / "ap4.sqlite")
    fresh = content_cycle(_FakeClient(_league_boxes()), 2025, 7, store=store,
                          notifier=None, per_week=2, generate=True)
    assert fresh and all(store.get(p.id).status.value == "proposed" for p in fresh)
    store.close()
