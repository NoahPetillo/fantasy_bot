"""Fernet credential encryption (hard requirement #2: encrypt cookies at rest)."""

from __future__ import annotations

import pytest

from fantasy.config import settings
from fantasy.security import crypto


@pytest.fixture
def enc_key(monkeypatch):
    key = crypto.generate_key()
    monkeypatch.setattr(settings, "credential_enc_key", key)
    crypto._cipher_for.cache_clear()
    yield key
    crypto._cipher_for.cache_clear()


def test_round_trip(enc_key):
    secret = "AEB...long-espn_s2-cookie-value...=="
    token = crypto.encrypt(secret)
    assert token != secret  # actually encrypted
    assert secret not in token  # plaintext never appears in the ciphertext
    assert crypto.decrypt(token) == secret


def test_ciphertext_is_nondeterministic(enc_key):
    # Fernet embeds a random IV, so encrypting twice yields different tokens.
    assert crypto.encrypt("same") != crypto.encrypt("same")


def test_tampered_token_rejected(enc_key):
    token = crypto.encrypt("secret")
    tampered = token[:-2] + ("AA" if not token.endswith("AA") else "BB")
    with pytest.raises(crypto.InvalidToken):
        crypto.decrypt(tampered)


def test_missing_key_raises(monkeypatch):
    monkeypatch.setattr(settings, "credential_enc_key", None)
    crypto._cipher_for.cache_clear()
    with pytest.raises(crypto.EncryptionError):
        crypto.encrypt("secret")


def test_key_rotation_decrypts_old_ciphertext(monkeypatch):
    old = crypto.generate_key()
    new = crypto.generate_key()
    # Encrypt under the old key only.
    monkeypatch.setattr(settings, "credential_enc_key", old)
    crypto._cipher_for.cache_clear()
    token = crypto.encrypt("secret")
    # Rotate: new key first, old key retained for decrypt (MultiFernet).
    monkeypatch.setattr(settings, "credential_enc_key", f"{new},{old}")
    crypto._cipher_for.cache_clear()
    assert crypto.decrypt(token) == "secret"  # still readable after rotation


def test_redact_never_reveals_secret_or_prefix_or_length():
    secret = "espn_s2_ABCDEFGHIJKLMNOP"
    masked = crypto.redact(secret)
    assert secret not in masked
    assert "espn" not in masked  # no prefix leak (can't fingerprint the type)
    assert str(len(secret)) not in masked  # no exact length leak
    assert crypto.redact(None) == "<none>"
