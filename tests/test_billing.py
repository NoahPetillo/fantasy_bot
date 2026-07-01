"""Phase 5 — plan-based quotas + Stripe subscription sync.

Covers hard requirement #5 (per-user, plan-based chat quota), the league-count
gate, /api/billing/status, checkout-requires-config, and the webhook plan sync.
Stripe is never called — the webhook verifier is stubbed.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

import fantasy.chat.agent as chat_agent
from fantasy.api import billing_routes
from fantasy.billing import service
from fantasy.billing.plans import chat_daily_limit
from fantasy.config import settings
from fantasy.db.models import ChatUsage, User
from fantasy.db.repos import add_league


@pytest.fixture(autouse=True)
def _stub_chat(monkeypatch):
    # Chat answer is stubbed so /api/chat never hits the network.
    monkeypatch.setattr(chat_agent, "answer",
                        lambda q, ctx: {"answer": "ok", "tools_used": [], "mode": "test"})


def _seed_usage(db, user, count):
    db.add(ChatUsage(user_id=user.id, day=datetime.now(timezone.utc).date(), count=count))
    db.commit()


# ── chat quota ──────────────────────────────────────────────────────────────
def test_free_chat_quota_blocks_over_limit(webapp):
    user = webapp.make_user("owner")
    _seed_usage(webapp.db, user, chat_daily_limit("free"))  # already at the cap
    webapp.auth_as(user)
    r = webapp.client.post("/api/chat", json={"question": "hi"})
    assert r.status_code == 429 and "daily limit" in r.json()["detail"].lower()
    assert r.headers.get("X-Quota-Exceeded") == "1"


def test_chat_allowed_and_counts_up(webapp):
    user = webapp.make_user("owner")
    webapp.auth_as(user)
    assert webapp.client.post("/api/chat", json={"question": "hi"}).status_code == 200
    assert webapp.client.get("/api/billing/status").json()["chat"]["used"] == 1


def test_pro_plan_has_higher_quota(webapp):
    user = webapp.make_user("owner")
    user.plan = "pro"
    webapp.db.commit()
    _seed_usage(webapp.db, user, chat_daily_limit("free"))  # over FREE, under PRO
    webapp.auth_as(user)
    assert webapp.client.post("/api/chat", json={"question": "hi"}).status_code == 200


def test_quota_is_per_user(webapp):
    a = webapp.make_user("a")
    b = webapp.make_user("b")
    _seed_usage(webapp.db, a, chat_daily_limit("free"))  # A maxed
    webapp.auth_as(b)  # B is fresh
    assert webapp.client.post("/api/chat", json={"question": "hi"}).status_code == 200


# ── league-count gate ───────────────────────────────────────────────────────
def test_free_plan_league_limit(webapp, monkeypatch):
    import fantasy.api.app as api
    monkeypatch.setattr(api, "build_shell_for", lambda db, user, lg, week=None: {"ok": True})
    user = webapp.make_user("owner")
    add_league(webapp.db, user, espn_league_id=111, team_id=1, season=2025)  # 1st (at free cap)
    webapp.auth_as(user)
    r = webapp.client.post("/api/leagues", json={"league_id": 222, "team_id": 1, "season": 2025})
    assert r.status_code == 402 and "upgrade to pro" in r.json()["detail"].lower()
    # Re-adding the SAME league/season is an update, not a new one → allowed.
    assert webapp.client.post("/api/leagues",
                              json={"league_id": 111, "team_id": 2, "season": 2025}).status_code == 200


# ── billing status + checkout config gate ───────────────────────────────────
def test_billing_status_free_defaults(webapp):
    webapp.auth_as(webapp.make_user("owner"))
    s = webapp.client.get("/api/billing/status").json()
    assert s["plan"] == "free" and s["max_leagues"] == 1
    assert s["chat"]["limit"] == chat_daily_limit("free")
    assert s["billing_enabled"] is False and s["can_manage"] is False


def test_checkout_requires_stripe_config(webapp):
    webapp.auth_as(webapp.make_user("owner"))
    assert webapp.client.post("/api/billing/checkout").status_code == 503


# ── webhook plan sync ───────────────────────────────────────────────────────
def _pass_through_verify(monkeypatch):
    import json
    monkeypatch.setattr(service, "verify_webhook", lambda payload, sig: json.loads(payload))


def test_webhook_upgrades_then_cancels(webapp, monkeypatch):
    user = webapp.make_user("owner")
    uid = user.id
    _pass_through_verify(monkeypatch)

    upgrade = {"id": "evt_1", "type": "checkout.session.completed",
               "data": {"object": {"metadata": {"user_id": str(uid)},
                                   "customer": "cus_1", "subscription": "sub_1"}}}
    assert webapp.client.post("/api/stripe/webhook", json=upgrade).status_code == 200
    webapp.db.expire_all()
    assert webapp.db.get(User, uid).plan == "pro"

    cancel = {"id": "evt_2", "type": "customer.subscription.deleted",
              "data": {"object": {"customer": "cus_1", "id": "sub_1"}}}
    assert webapp.client.post("/api/stripe/webhook", json=cancel).status_code == 200
    webapp.db.expire_all()
    assert webapp.db.get(User, uid).plan == "free"


def test_webhook_is_idempotent_on_replay(webapp, monkeypatch):
    user = webapp.make_user("owner")
    uid = user.id
    _pass_through_verify(monkeypatch)
    upgrade = {"id": "evt_dupe", "type": "checkout.session.completed",
               "data": {"object": {"metadata": {"user_id": str(uid)}, "customer": "cus_9",
                                   "subscription": "sub_9"}}}
    assert webapp.client.post("/api/stripe/webhook", json=upgrade).json() == {"received": True}
    # Manually downgrade, then REPLAY the same event id — dedup must NOT re-upgrade.
    webapp.db.expire_all()  # see the request session's committed 'pro' first
    u = webapp.db.get(User, uid); u.plan = "free"; webapp.db.commit()
    r = webapp.client.post("/api/stripe/webhook", json=upgrade).json()
    assert r.get("duplicate") is True
    webapp.db.expire_all()
    assert webapp.db.get(User, uid).plan == "free"  # replay was a no-op


def test_webhook_bad_signature_rejected(webapp, monkeypatch):
    monkeypatch.setattr(service, "verify_webhook",
                        lambda payload, sig: (_ for _ in ()).throw(ValueError("bad signature")))
    assert billing_routes  # imported
    assert webapp.client.post("/api/stripe/webhook", json={"type": "x"}).status_code == 400


def test_webhook_real_signature_and_stripe_event_object(webapp, monkeypatch):
    """End-to-end WITHOUT stubbing verify_webhook: a real Stripe-signed payload runs
    through stripe.Webhook.construct_event. Guards against assuming the event is a
    plain dict (the SDK returns a StripeObject)."""
    import hashlib
    import hmac
    import json
    import time

    secret = "whsec_testsecret_abc123"
    monkeypatch.setattr(settings, "stripe_secret_key", "sk_test_dummy")
    monkeypatch.setattr(settings, "stripe_webhook_secret", secret)
    user = webapp.make_user("owner")
    uid = user.id
    payload = json.dumps({
        "id": "evt_real", "object": "event", "type": "checkout.session.completed",
        "data": {"object": {"metadata": {"user_id": str(uid)},
                            "customer": "cus_real", "subscription": "sub_real"}}})
    ts = int(time.time())
    sig = hmac.new(secret.encode(), f"{ts}.{payload}".encode(), hashlib.sha256).hexdigest()
    r = webapp.client.post("/api/stripe/webhook", content=payload,
                           headers={"stripe-signature": f"t={ts},v1={sig}",
                                    "content-type": "application/json"})
    assert r.status_code == 200, r.text
    webapp.db.expire_all()
    assert webapp.db.get(User, uid).plan == "pro"


def test_webhook_fails_closed_without_secret(webapp):
    # No monkeypatch: real verify_webhook runs; STRIPE_* is nulled by the hermetic
    # fixture, so it must fail closed (503), never silently accept.
    r = webapp.client.post("/api/stripe/webhook", json={"id": "evt", "type": "x"})
    assert r.status_code == 503
