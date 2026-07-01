"""Serve the legal/consent copy from the drafts in ``legal/``.

The ESPN connect-consent copy is the load-bearing legal/UX surface (see
MULTITENANT_BUILD.md): it must be shown at the connect step behind a required,
unchecked-by-default checkbox, and the accepted **version** is persisted with the
credentials. Bump ``ESPN_CONSENT_VERSION`` whenever the copy materially changes so
we can tell who agreed to what.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fantasy.config import settings

# Bump when the consent copy below materially changes.
ESPN_CONSENT_VERSION = "1.0"

_LEGAL_DIR = Path(__file__).resolve().parent.parent / "legal"


def _read(name: str) -> str:
    return (_LEGAL_DIR / name).read_text(encoding="utf-8")


def _body_after_frontmatter(md: str) -> str:
    """Drop the leading ``>`` editor note / instructions above the first ``---``."""
    marker = "\n---\n"
    idx = md.find(marker)
    return md[idx + len(marker):].strip() if idx != -1 else md.strip()


def _fill(md: str) -> str:
    return md.replace("[PRODUCT_NAME]", settings.product_name)


@lru_cache(maxsize=1)
def espn_consent_markdown() -> str:
    """The consent copy to render at the Connect-ESPN step (product name filled in,
    editor preamble stripped)."""
    return _fill(_body_after_frontmatter(_read("ESPN_CONNECT_CONSENT.md")))


@lru_cache(maxsize=2)
def policy_markdown(kind: str) -> str:
    """Privacy / Terms copy (served at /privacy and /terms in Phase 4)."""
    fname = {"privacy": "PRIVACY.md", "terms": "TERMS.md"}[kind]
    return _fill(_read(fname))
