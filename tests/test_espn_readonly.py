"""Read-only to ESPN, always (hard requirement #3).

Two guarantees:
1. Static — the ESPN client's source contains no HTTP write verbs; it can only GET.
2. Runtime — a per-user client built from that user's cookies issues a GET with
   exactly those cookies, and never a POST/PUT/PATCH/DELETE.
"""

from __future__ import annotations

import inspect
import re

import pytest

from fantasy.espn import client as espn_client
from fantasy.espn.client import EspnClient

_WRITE_VERBS = ("post", "put", "patch", "delete")


def test_client_source_has_no_http_write_verbs():
    from fantasy.espn import account as espn_account

    for mod in (espn_client, espn_account):
        src = inspect.getsource(mod)
        for verb in _WRITE_VERBS:
            # e.g. requests.post( / session.put( / .patch(
            assert not re.search(rf"\.{verb}\s*\(", src), \
                f"{mod.__name__} must not call .{verb}()"
        assert re.search(r"requests\.get\s*\(", src), f"expected {mod.__name__} to use requests.get"


def test_client_exposes_no_write_methods():
    write_like = re.compile(
        r"^(set_|add_|drop_|submit_|propose_|execute|write|update_|delete_|post_|put_|claim_|trade_)"
    )
    offenders = [
        name for name, _ in inspect.getmembers(EspnClient, predicate=inspect.isfunction)
        if not name.startswith("_") and write_like.match(name)
    ]
    assert offenders == [], f"EspnClient exposes write-looking methods: {offenders}"


class _FakeResp:
    status_code = 200

    def json(self):
        return {"settings": {}}

    def raise_for_status(self):
        return None


def test_per_user_client_uses_only_get_with_that_users_cookies(monkeypatch):
    captured = {}

    def fake_get(url, params=None, cookies=None, headers=None, timeout=None):
        captured["url"] = url
        captured["cookies"] = cookies
        return _FakeResp()

    def forbidden(*a, **k):  # any write verb reaching the network is a hard failure
        raise AssertionError("ESPN client attempted a write request")

    monkeypatch.setattr(espn_client.requests, "get", fake_get)
    for verb in _WRITE_VERBS:
        monkeypatch.setattr(espn_client.requests, verb, forbidden, raising=False)

    # Per-user client: cookies passed explicitly (as the multi-tenant request path
    # will, after decrypting them), NOT read from global settings.
    c = EspnClient(league_id=42, season=2025, espn_s2="user_s2_value", swid="{USER-SWID}")
    c._raw(["mSettings"])

    assert captured["cookies"] == {"espn_s2": "user_s2_value", "SWID": "{USER-SWID}"}
    assert "/leagues/42" in captured["url"]
