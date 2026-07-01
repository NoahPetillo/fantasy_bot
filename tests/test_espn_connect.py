"""Phase 2 — Connect ESPN: consent, encryption at rest, test/purge, deletion.

Covers hard requirements #2 (cookies encrypted, never returned, delete endpoints,
auto-purge) and #4 (consent before storage), plus per-user isolation.
"""

from __future__ import annotations

import pytest
from fastapi import Depends
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import fantasy.api.app as api
from fantasy.api import espn_routes
from fantasy.api.clerk_auth import get_current_user
from fantasy.config import settings
from fantasy.db.base import get_db
from fantasy.db.models import EspnCredential, User
from fantasy.espn import credentials as creds
from fantasy.security import crypto

S2 = "AEB_real_looking_espn_s2_cookie_value_1234567890"
SWID_RAW = "ABCD1234-EF56-7890-ABCD-1234567890AB"


@pytest.fixture(autouse=True)
def enc_key(monkeypatch):
    monkeypatch.setattr(settings, "credential_enc_key", crypto.generate_key())
    crypto._cipher_for.cache_clear()
    yield
    crypto._cipher_for.cache_clear()


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    api.app.dependency_overrides.clear()


def _mk_user(db, clerk_id):
    u = User(clerk_user_id=clerk_id, email=f"{clerk_id}@ex.com")
    db.add(u)
    db.commit()
    db.refresh(u)
    return u


@pytest.fixture
def user(db):
    return _mk_user(db, "u_main")


def _client_for(uid) -> TestClient:
    def _current(dbs: Session = Depends(get_db)):
        return dbs.get(User, uid)
    api.app.dependency_overrides[get_current_user] = _current
    return TestClient(api.app)


@pytest.fixture
def client(user):
    return _client_for(user.id)


@pytest.fixture
def espn_ok(monkeypatch):
    monkeypatch.setattr(espn_routes, "validate_cookies", lambda s2, swid: True)
    monkeypatch.setattr(espn_routes, "discover_ff_leagues",
                        lambda s2, swid: [{"league_id": 5, "team_id": 1, "season": 2025, "name": "Dynasty"}])


# ── consent gate (hard requirement #4) ───────────────────────────────────────
def test_connect_blocked_without_consent(client, db, user, espn_ok):
    r = client.post("/api/espn/connect", json={"espn_s2": S2, "swid": SWID_RAW})  # consent omitted
    assert r.status_code == 400
    db.expire_all()
    assert db.get(EspnCredential, user.id) is None  # nothing stored


def test_connect_requires_both_cookies(client, espn_ok):
    r = client.post("/api/espn/connect", json={"consent": True, "espn_s2": S2})
    assert r.status_code == 400


# ── encryption at rest + consent persisted (hard requirement #2 & #4) ─────────
def test_connect_stores_encrypted_and_persists_consent(client, db, user, espn_ok):
    r = client.post("/api/espn/connect", json={"consent": True, "espn_s2": S2, "swid": SWID_RAW})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["connected"] is True and body["leagues_found"]

    db.expire_all()
    c = db.get(EspnCredential, user.id)
    assert c is not None
    # Stored ciphertext, not plaintext.
    assert c.s2_enc != S2 and S2 not in c.s2_enc
    assert SWID_RAW not in c.swid_enc
    # Round-trips; SWID stored normalized (braced).
    assert crypto.decrypt(c.s2_enc) == S2
    assert crypto.decrypt(c.swid_enc) == "{" + SWID_RAW + "}"
    # Consent recorded.
    assert c.consent_version == creds.ESPN_CONSENT_VERSION and c.consent_at is not None


def test_connect_response_never_leaks_cookies(client, espn_ok):
    r = client.post("/api/espn/connect", json={"consent": True, "espn_s2": S2, "swid": SWID_RAW})
    assert S2 not in r.text and SWID_RAW not in r.text


def test_connect_rejects_invalid_cookies(client, db, user, monkeypatch):
    monkeypatch.setattr(espn_routes, "validate_cookies", lambda s2, swid: False)
    r = client.post("/api/espn/connect", json={"consent": True, "espn_s2": "bad", "swid": "bad"})
    assert r.status_code == 400
    db.expire_all()
    assert db.get(EspnCredential, user.id) is None


def test_connect_transient_espn_error_does_not_store(client, db, user, monkeypatch):
    def boom(s2, swid):
        raise RuntimeError("network down")
    monkeypatch.setattr(espn_routes, "validate_cookies", boom)
    r = client.post("/api/espn/connect", json={"consent": True, "espn_s2": S2, "swid": SWID_RAW})
    assert r.status_code == 502
    db.expire_all()
    assert db.get(EspnCredential, user.id) is None


# ── status + test connection ─────────────────────────────────────────────────
def test_status_before_and_after(client, espn_ok):
    assert client.get("/api/espn/status").json() == {"connected": False}
    client.post("/api/espn/connect", json={"consent": True, "espn_s2": S2, "swid": SWID_RAW})
    s = client.get("/api/espn/status").json()
    assert s["connected"] is True and s["consent_version"] == creds.ESPN_CONSENT_VERSION
    assert S2 not in str(s) and SWID_RAW not in str(s)  # status never exposes cookies


def test_test_connection_valid(client, espn_ok):
    client.post("/api/espn/connect", json={"consent": True, "espn_s2": S2, "swid": SWID_RAW})
    r = client.post("/api/espn/test").json()
    assert r["valid"] is True and r["leagues_found"]


def test_test_connection_auto_purges_after_repeated_failures(client, db, user, espn_ok, monkeypatch):
    client.post("/api/espn/connect", json={"consent": True, "espn_s2": S2, "swid": SWID_RAW})
    # Now cookies go stale.
    monkeypatch.setattr(espn_routes, "validate_cookies", lambda s2, swid: False)
    for i in range(1, creds.MAX_AUTH_FAILURES):
        r = client.post("/api/espn/test").json()
        assert r["valid"] is False and r["purged"] is False
        assert client.get("/api/espn/status").json()["failure_count"] == i
    # Threshold hit → purged.
    r = client.post("/api/espn/test").json()
    assert r["valid"] is False and r["purged"] is True
    db.expire_all()
    assert db.get(EspnCredential, user.id) is None
    # Subsequent test has nothing to check.
    assert client.post("/api/espn/test").status_code == 404


# ── deletion (immediate) ─────────────────────────────────────────────────────
def test_delete_credentials(client, db, user, espn_ok):
    client.post("/api/espn/connect", json={"consent": True, "espn_s2": S2, "swid": SWID_RAW})
    assert client.delete("/api/espn/credentials").json()["ok"] is True
    db.expire_all()
    assert db.get(EspnCredential, user.id) is None
    assert client.get("/api/espn/status").json() == {"connected": False}


def test_delete_account_removes_user_and_credentials(client, db, user, espn_ok):
    client.post("/api/espn/connect", json={"consent": True, "espn_s2": S2, "swid": SWID_RAW})
    uid = user.id
    assert client.delete("/api/account").json()["deleted"] is True
    db.expire_all()
    assert db.get(User, uid) is None
    assert db.get(EspnCredential, uid) is None


# ── per-user isolation ───────────────────────────────────────────────────────
def test_isolation_between_users(db, espn_ok):
    a = _mk_user(db, "u_a")
    b = _mk_user(db, "u_b")
    _client_for(a.id).post("/api/espn/connect", json={"consent": True, "espn_s2": S2, "swid": SWID_RAW})
    # B is unaffected by A connecting.
    assert _client_for(b.id).get("/api/espn/status").json() == {"connected": False}
    assert _client_for(b.id).post("/api/espn/test").status_code == 404
    # A remains connected.
    assert _client_for(a.id).get("/api/espn/status").json()["connected"] is True


# ── per-user EspnClient wiring (read-only, from the user's own cookies) ───────
def test_build_client_for_user_uses_that_users_cookies(db, user):
    creds.store_credentials(db, user, S2, SWID_RAW, consent_version="1.0")
    c = creds.build_client_for_user(db, user, league_id=77, season=2024)
    assert c.league_id == 77 and c.season == 2024
    assert c.espn_s2 == S2 and c.swid == "{" + SWID_RAW + "}"
    assert c.cookies == {"espn_s2": S2, "SWID": "{" + SWID_RAW + "}"}


def test_build_client_requires_connection(db, user):
    from fantasy.espn.client import EspnAuthError
    with pytest.raises(EspnAuthError):
        creds.build_client_for_user(db, user, league_id=1)


# ── discovery hardening: ids are coerced to ints, junk entries dropped ────────
def test_discover_coerces_ids_and_drops_injection(monkeypatch):
    from fantasy.espn import account
    profile = {"preferences": [
        {"metaData": {"entry": {"gameId": "ffl", "seasonId": "2025", "entryId": "7",
                                "groups": [{"groupId": "42", "groupName": "Good"}]}}},
        # A hostile season/league id must not survive as a string (int() drops it).
        {"metaData": {"entry": {"gameId": "ffl", "seasonId": "<script>", "entryId": "1",
                                "groups": [{"groupId": "<img onerror=x>", "groupName": "Bad"}]}}},
    ]}
    monkeypatch.setattr(account, "fetch_fan_profile", lambda s2, swid: profile)
    out = account.discover_ff_leagues("s2", "swid")
    assert out == [{"league_id": 42, "team_id": 7, "season": 2025, "name": "Good"}]
    for lg in out:  # nothing an attacker could inject reaches the caller as a string
        assert isinstance(lg["league_id"], int)
        assert lg["season"] is None or isinstance(lg["season"], int)


# ── consent copy endpoint (public disclosure) ────────────────────────────────
def test_consent_copy_public(db):
    # No auth override — this route is public.
    r = TestClient(api.app).get("/api/legal/espn-consent")
    assert r.status_code == 200
    body = r.json()
    assert body["version"] == creds.ESPN_CONSENT_VERSION
    assert "[PRODUCT_NAME]" not in body["markdown"]  # placeholder filled
    assert settings.product_name in body["markdown"]
    md = body["markdown"].lower()
    assert "we only ever read" in md  # the read-only promise
    assert "not affiliated" in md  # the ESPN disclaimer
