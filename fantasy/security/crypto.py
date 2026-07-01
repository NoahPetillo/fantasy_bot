"""Fernet encryption for ESPN session cookies (and any other secret at rest).

The key comes from ``CREDENTIAL_ENC_KEY`` and is held in the host's secret store
— **never in the database**. A database leak alone therefore cannot expose any
user's cookies (hard requirement #2, and the promise in legal/PRIVACY.md §3).

Key rotation: ``CREDENTIAL_ENC_KEY`` may be a comma-separated list. The first key
encrypts new values; every key is tried on decrypt (``MultiFernet``), so you can
add a new key, re-encrypt over time, then drop the old one.

Generate a key:  ``python -m fantasy.security.crypto``
"""

from __future__ import annotations

from functools import lru_cache

from cryptography.fernet import Fernet, InvalidToken, MultiFernet

from fantasy.config import settings

__all__ = ["encrypt", "decrypt", "generate_key", "redact", "EncryptionError", "InvalidToken"]


class EncryptionError(RuntimeError):
    """Raised when encryption is requested but no key is configured."""


@lru_cache(maxsize=1)
def _cipher_for(raw_key: str) -> MultiFernet:
    keys = [Fernet(k.strip().encode()) for k in raw_key.split(",") if k.strip()]
    if not keys:
        raise EncryptionError("CREDENTIAL_ENC_KEY is empty.")
    return MultiFernet(keys)


def _cipher() -> MultiFernet:
    raw = settings.credential_enc_key
    if not raw:
        raise EncryptionError(
            "CREDENTIAL_ENC_KEY is not set — cannot encrypt/decrypt credentials. "
            "Generate one with `python -m fantasy.security.crypto`."
        )
    return _cipher_for(raw)


def encrypt(plaintext: str) -> str:
    """Encrypt a UTF-8 string, returning a URL-safe base64 token."""
    return _cipher().encrypt(plaintext.encode()).decode()


def decrypt(token: str) -> str:
    """Decrypt a token produced by :func:`encrypt`. Raises ``InvalidToken`` if the
    ciphertext was tampered with or no configured key can decrypt it."""
    return _cipher().decrypt(token.encode()).decode()


def generate_key() -> str:
    """A fresh Fernet key (base64 str) suitable for ``CREDENTIAL_ENC_KEY``."""
    return Fernet.generate_key().decode()


def redact(value: str | None) -> str:
    """Mask a secret for safe display in logs/errors.

    Reveals nothing identifying — not the value, not its prefix, and not its exact
    length (only a coarse size bucket), so credential types can't be fingerprinted
    from logs.
    """
    if not value:
        return "<none>"
    bucket = (len(value) // 16) * 16
    return f"<redacted:{bucket}-{bucket + 16} chars>"


if __name__ == "__main__":  # pragma: no cover
    print(generate_key())
