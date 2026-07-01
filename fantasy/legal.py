"""Serve the legal/consent copy from the drafts in ``legal/``.

The ESPN connect-consent copy is the load-bearing legal/UX surface (see
MULTITENANT_BUILD.md): it must be shown at the connect step behind a required,
unchecked-by-default checkbox, and the accepted **version** is persisted with the
credentials. Bump ``ESPN_CONSENT_VERSION`` whenever the copy materially changes so
we can tell who agreed to what.
"""

from __future__ import annotations

import html
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


def _fill(md: str, product: str) -> str:
    return md.replace("[PRODUCT_NAME]", product)


def espn_consent_markdown() -> str:
    """The consent copy to render at the Connect-ESPN step (product name filled in,
    editor preamble stripped). Returned as markdown; the client escapes on render,
    so the raw product name is safe here."""
    body = _body_after_frontmatter(_read("ESPN_CONNECT_CONSENT.md"))
    return _fill(body, settings.product_name)


def policy_markdown(kind: str) -> str:
    """Privacy / Terms copy. Rendered to HTML server-side with raw-HTML passthrough,
    so the one dynamic value (product name) is HTML-escaped; the .md files
    themselves are trusted, version-controlled content."""
    fname = {"privacy": "PRIVACY.md", "terms": "TERMS.md"}[kind]
    return _fill(_read(fname), html.escape(settings.product_name))


_PAGE = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ margin: 0; background: #0f1115; color: #e7ebf0;
    font: 16px/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}
  main {{ max-width: 760px; margin: 0 auto; padding: 48px 22px 24px; }}
  h1 {{ font-size: 26px; }} h2 {{ font-size: 20px; margin-top: 32px; }}
  a {{ color: #4c9ffe; }} code {{ background: #1b1f27; padding: 1px 5px; border-radius: 5px; }}
  blockquote {{ border-left: 3px solid #2f7de0; margin: 16px 0; padding: 4px 16px; color: #9aa4b2; }}
  hr {{ border: 0; border-top: 1px solid #262b36; margin: 28px 0; }}
  footer {{ max-width: 760px; margin: 0 auto; padding: 24px 22px 56px; color: #7c8695;
    font-size: 13px; border-top: 1px solid #262b36; }}
</style></head><body>
<main>{body}</main>
<footer>{product} is an independent tool, <strong>not affiliated with, authorized, or
endorsed by ESPN or The Walt Disney Company.</strong><br>
<a href="/">Home</a> · <a href="/privacy">Privacy</a> · <a href="/terms">Terms</a></footer>
</body></html>"""


def render_policy_html(kind: str) -> str:
    """Server-rendered HTML for /privacy and /terms (with the ESPN disclaimer footer)."""
    import markdown as _md

    title = {"privacy": "Privacy Policy", "terms": "Terms of Service"}[kind]
    body = _md.markdown(policy_markdown(kind), extensions=["extra", "sane_lists"])
    safe = html.escape(settings.product_name)  # never inject raw config into HTML
    return _PAGE.format(title=f"{title} — {safe}", body=body, product=safe)
