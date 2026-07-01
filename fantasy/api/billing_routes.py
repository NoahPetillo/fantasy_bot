"""Billing endpoints (Phase 5): plan/usage status, Stripe checkout + portal, and
the Stripe webhook. The webhook is public (Stripe calls it) and verified by
signature; everything else is Clerk-scoped to the current user."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from fantasy.api.clerk_auth import get_current_user
from fantasy.billing import quota, service
from fantasy.billing.plans import label, max_leagues
from fantasy.config import settings
from fantasy.db.base import get_db
from fantasy.db.models import User
from fantasy.db.repos import list_leagues

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["billing"])


@router.get("/billing/status")
def billing_status(user: User = Depends(get_current_user),
                   db: Session = Depends(get_db)) -> dict:
    sub = service.get_subscription(db, user)
    return {
        "plan": user.plan,
        "plan_label": label(user.plan),
        "chat": quota.chat_status(db, user),
        "leagues_used": len(list_leagues(db, user)),
        "max_leagues": max_leagues(user.plan),
        "billing_enabled": bool(settings.stripe_secret_key and settings.stripe_price_id),
        "can_manage": bool(sub and sub.stripe_customer_id),
    }


@router.post("/billing/checkout")
def billing_checkout(request: Request, user: User = Depends(get_current_user),
                     db: Session = Depends(get_db)) -> dict:
    """Create a Stripe Checkout session for the Pro plan; returns the redirect URL."""
    try:
        url = service.create_checkout_session(db, user, str(request.base_url).rstrip("/"))
    except service.BillingNotConfigured as e:
        raise HTTPException(503, str(e))
    return {"url": url}


@router.post("/billing/portal")
def billing_portal(request: Request, user: User = Depends(get_current_user),
                   db: Session = Depends(get_db)) -> dict:
    """Create a Stripe Billing Portal session (manage/cancel); returns the URL."""
    try:
        url = service.create_portal_session(db, user, str(request.base_url).rstrip("/"))
    except service.BillingNotConfigured as e:
        raise HTTPException(503, str(e))
    return {"url": url}


@router.post("/stripe/webhook")
async def stripe_webhook(request: Request, db: Session = Depends(get_db)) -> dict:
    """Stripe → us. Verifies the signature, then syncs plan/subscription state.
    Public (no Clerk); trust comes from the signature."""
    payload = await request.body()
    sig = request.headers.get("stripe-signature", "")
    try:
        service.verify_webhook(payload, sig)  # signature check (raises on bad/unconfigured)
    except service.BillingNotConfigured as e:
        raise HTTPException(503, str(e))
    except Exception:  # noqa: BLE001 — bad/forged signature
        raise HTTPException(400, "invalid signature")
    # Parse the (now signature-verified) raw bytes as a plain dict — the Stripe SDK
    # returns an Event object whose field access differs; a dict keeps .get() simple.
    event = json.loads(payload)
    # Idempotency: Stripe retries/replays events; skip ones we've already applied.
    event_id = event.get("id")
    if event_id and service.already_processed(db, event_id):
        return {"received": True, "duplicate": True}
    # A transient DB error here PROPAGATES to a 500 on purpose, so Stripe retries
    # (returning 200 would drop the event). Unknown users / event types are no-ops.
    service.apply_event(db, event)
    if event_id:
        service.mark_processed(db, event_id)
    return {"received": True}
