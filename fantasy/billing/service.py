"""Stripe subscription service: customer / checkout / portal + webhook sync.

Keeps ``users.plan`` (the fast quota-check field) and the ``subscriptions`` row in
sync with Stripe. All Stripe access goes through ``_stripe()`` so the app runs
fine with billing unconfigured (checkout/portal simply 503).
"""

from __future__ import annotations

import logging
import uuid as _uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from fantasy.billing.plans import FREE, PRO
from fantasy.config import settings
from fantasy.db.models import Subscription, User, WebhookEvent

log = logging.getLogger(__name__)

_ACTIVE_STATUSES = {"active", "trialing"}


class BillingNotConfigured(RuntimeError):
    """Raised when a Stripe operation is attempted without the required config."""


def _stripe():
    if not settings.stripe_secret_key:
        raise BillingNotConfigured("Billing isn't configured (STRIPE_SECRET_KEY unset).")
    import stripe

    stripe.api_key = settings.stripe_secret_key
    return stripe


def get_subscription(db: Session, user: User) -> Subscription | None:
    return db.get(Subscription, user.id)


def _sub_row(db: Session, user: User) -> Subscription:
    sub = db.get(Subscription, user.id)
    if sub is None:
        sub = Subscription(user_id=user.id, plan=user.plan or FREE)
        db.add(sub)
        db.commit()
        db.refresh(sub)
    return sub


def ensure_customer(db: Session, user: User) -> str:
    stripe = _stripe()
    sub = _sub_row(db, user)
    if sub.stripe_customer_id:
        return sub.stripe_customer_id
    customer = stripe.Customer.create(email=user.email or None,
                                      metadata={"user_id": str(user.id)})
    sub.stripe_customer_id = customer.id
    db.commit()
    return customer.id


def create_checkout_session(db: Session, user: User, base_url: str) -> str:
    if not settings.stripe_price_id:
        raise BillingNotConfigured("Billing isn't configured (STRIPE_PRICE_ID unset).")
    stripe = _stripe()
    customer_id = ensure_customer(db, user)
    session = stripe.checkout.Session.create(
        mode="subscription",
        customer=customer_id,
        line_items=[{"price": settings.stripe_price_id, "quantity": 1}],
        success_url=f"{base_url}/?billing=success",
        cancel_url=f"{base_url}/?billing=cancel",
        client_reference_id=str(user.id),
        metadata={"user_id": str(user.id)},
        allow_promotion_codes=True,
    )
    return session.url


def create_portal_session(db: Session, user: User, base_url: str) -> str:
    stripe = _stripe()
    sub = get_subscription(db, user)
    if not (sub and sub.stripe_customer_id):
        raise BillingNotConfigured("No Stripe customer for this account yet.")
    session = stripe.billing_portal.Session.create(
        customer=sub.stripe_customer_id, return_url=f"{base_url}/")
    return session.url


def verify_webhook(payload: bytes, sig_header: str):
    stripe = _stripe()
    if not settings.stripe_webhook_secret:
        raise BillingNotConfigured("Billing isn't configured (STRIPE_WEBHOOK_SECRET unset).")
    return stripe.Webhook.construct_event(payload, sig_header, settings.stripe_webhook_secret)


def already_processed(db: Session, event_id: str) -> bool:
    """True if this Stripe event id was already applied (webhook idempotency)."""
    return db.get(WebhookEvent, event_id) is not None


def mark_processed(db: Session, event_id: str) -> None:
    db.add(WebhookEvent(event_id=event_id))
    try:
        db.commit()
    except IntegrityError:  # concurrent delivery recorded it first
        db.rollback()


# ── webhook → local state sync ────────────────────────────────────────────────
def _find_user(db: Session, *, user_id=None, customer_id=None) -> User | None:
    if user_id:
        try:
            u = db.get(User, _uuid.UUID(str(user_id)))
        except (ValueError, TypeError):
            u = None
        if u:
            return u
    if customer_id:
        sub = db.execute(
            select(Subscription).where(Subscription.stripe_customer_id == customer_id)
        ).scalar_one_or_none()
        if sub:
            return db.get(User, sub.user_id)
    return None


def _set_plan(db: Session, user: User, plan: str, *, status=None, customer_id=None,
              subscription_id=None, period_end=None) -> None:
    user.plan = plan
    sub = _sub_row(db, user)
    sub.plan = plan
    if status is not None:
        sub.status = status
    if customer_id:
        sub.stripe_customer_id = customer_id
    if subscription_id:
        sub.stripe_subscription_id = subscription_id
    if period_end:
        sub.current_period_end = datetime.fromtimestamp(int(period_end), tz=timezone.utc)
    db.commit()
    log.info("Billing: user %s plan -> %s (status=%s)", user.id, plan, status)


def apply_event(db: Session, event) -> None:
    """Update local plan/subscription from a verified Stripe event. Unknown event
    types are ignored."""
    etype = event["type"]
    obj = event["data"]["object"]
    meta = obj.get("metadata") or {}

    if etype == "checkout.session.completed":
        user = _find_user(db, user_id=meta.get("user_id") or obj.get("client_reference_id"),
                          customer_id=obj.get("customer"))
        if user:
            _set_plan(db, user, PRO, status="active", customer_id=obj.get("customer"),
                      subscription_id=obj.get("subscription"))
    elif etype in ("customer.subscription.created", "customer.subscription.updated"):
        user = _find_user(db, customer_id=obj.get("customer"), user_id=meta.get("user_id"))
        if user:
            active = obj.get("status") in _ACTIVE_STATUSES
            _set_plan(db, user, PRO if active else FREE, status=obj.get("status"),
                      customer_id=obj.get("customer"), subscription_id=obj.get("id"),
                      period_end=obj.get("current_period_end"))
    elif etype == "customer.subscription.deleted":
        user = _find_user(db, customer_id=obj.get("customer"), user_id=meta.get("user_id"))
        if user:
            _set_plan(db, user, FREE, status="canceled", customer_id=obj.get("customer"),
                      subscription_id=obj.get("id"))
    else:
        log.info("Billing: ignoring Stripe event %s", etype)
