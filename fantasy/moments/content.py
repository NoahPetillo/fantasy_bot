"""Generate the postable assets for a moment: a caption and a square graphic.

Two reliability principles, both straight out of the research:

1. **Scores/names are real text, never model-generated.** AI image models can't be
   trusted to render exact numbers, so the graphic is templated (Playwright→PNG,
   with a Pillow fallback) and the LLM only writes the *caption*.
2. **Everything degrades gracefully.** No ANTHROPIC key → templated caption. No
   browser for Playwright → Pillow card. Pillow somehow unavailable → no image,
   caption still posts. The pipeline never hard-fails on a missing optional dep.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

from fantasy.config import settings
from fantasy.moments.models import Moment, MomentType

log = logging.getLogger(__name__)

MODEL = "claude-haiku-4-5-20251001"  # matches fantasy/news/extract.py

_ACCENT = {
    MomentType.nailbiter: "#ff5d3b",
    MomentType.blowout: "#9b5de5",
    MomentType.high_score: "#2ec4b6",
    MomentType.low_score: "#5b6b8c",
    MomentType.bench_blunder: "#f4a259",
    MomentType.lucky: "#ffd23f",
    MomentType.unlucky: "#ef476f",
    MomentType.boom: "#06d6a0",
    MomentType.bust: "#8d99ae",
    MomentType.hot_streak: "#ff7847",
    MomentType.cold_streak: "#4cc9f0",
    MomentType.rivalry: "#e63946",
    MomentType.trade: "#2ec4b6",
    MomentType.waiver: "#ffd23f",
}
_EMOJI = {
    MomentType.nailbiter: "😱", MomentType.blowout: "💀", MomentType.high_score: "👑",
    MomentType.low_score: "🚮", MomentType.bench_blunder: "🪑", MomentType.lucky: "🍀",
    MomentType.unlucky: "🥲", MomentType.boom: "🚀", MomentType.bust: "🥚",
    MomentType.hot_streak: "🔥", MomentType.cold_streak: "🥶", MomentType.rivalry: "😤",
    MomentType.trade: "🤝", MomentType.waiver: "💰",
}
_LABEL = {
    MomentType.nailbiter: "NAIL-BITER", MomentType.blowout: "BLOWOUT",
    MomentType.high_score: "TOP SCORE", MomentType.low_score: "LOW SCORE",
    MomentType.bench_blunder: "BENCH BLUNDER", MomentType.lucky: "LUCKY W",
    MomentType.unlucky: "UNLUCKY L", MomentType.boom: "BOOM", MomentType.bust: "BUST",
    MomentType.hot_streak: "HOT STREAK", MomentType.cold_streak: "COLD STREAK",
    MomentType.rivalry: "RIVALRY", MomentType.trade: "TRADE ALERT", MomentType.waiver: "FAAB SPLASH",
}


# ── caption ──────────────────────────────────────────────────────────────────
def write_caption(moment: Moment) -> str:
    """A short, punchy caption. LLM if a key is configured, else a clean template."""
    if not settings.anthropic_api_key:
        return _fallback_caption(moment)
    try:
        from anthropic import Anthropic

        client = Anthropic(api_key=settings.anthropic_api_key)
        resp = client.messages.create(
            model=MODEL, max_tokens=180,
            messages=[{"role": "user", "content": _caption_prompt(moment)}],
        )
        text = "".join(
            getattr(b, "text", "") for b in resp.content if getattr(b, "type", None) == "text"
        ).strip()
        return text or _fallback_caption(moment)
    except Exception as e:  # noqa: BLE001
        log.warning("Caption LLM failed (%s); falling back to template.", e)
        return _fallback_caption(moment)


def _caption_prompt(moment: Moment) -> str:
    league = settings.content_league_name or "our fantasy league"
    voice = (
        "Write 1–2 sentences of playful, savage group-chat trash talk. Emojis are great. "
        "Do NOT use hashtags."
        if settings.content_voice != "instagram"
        else "Write a punchy Instagram caption (1–2 sentences) with a little trash talk, "
        "then 3–5 relevant hashtags on a new line."
    )
    return (
        f"You are the smack-talking commissioner-bot for {league}. {voice}\n"
        f"Use ONLY these facts — do not invent any names or numbers, and keep names/scores exact:\n"
        f"FACTS: {moment.blurb}\n"
        f"Return only the caption text, nothing else."
    )


def _fallback_caption(moment: Moment) -> str:
    emoji = _EMOJI.get(moment.type, "🔥")
    base = f"{emoji} {moment.blurb}"
    if settings.content_voice == "instagram":
        tags = {
            MomentType.nailbiter: "#nailbiter #fantasyfootball #closegame",
            MomentType.blowout: "#blowout #fantasyfootball #domination",
            MomentType.bench_blunder: "#benchwoes #fantasyfootball #coachingmalpractice",
            MomentType.unlucky: "#unlucky #fantasyfootball #robbed",
            MomentType.lucky: "#luckywin #fantasyfootball #scheduleluck",
            MomentType.boom: "#boom #fantasyfootball #league",
            MomentType.bust: "#bust #fantasyfootball #benchhim",
        }.get(moment.type, "#fantasyfootball #league")
        return f"{base}\n{tags}"
    return base


# ── graphic ──────────────────────────────────────────────────────────────────
def render_card(moment: Moment, out_dir: Path | None = None) -> Path | None:
    """Render a 1080×1080 PNG for the moment. Returns the path, or None if no
    renderer is available (caption still posts on its own)."""
    out_dir = out_dir or settings.media_dir
    h = hashlib.sha1(f"{moment.type.value}:{moment.dedup_key}".encode()).hexdigest()[:8]
    out = Path(out_dir) / f"wk{moment.week:02d}_{moment.type.value}_{h}.png"
    if _render_playwright(moment, out):
        return out
    if _render_pillow(moment, out):
        return out
    log.warning("No graphic renderer available for moment %s; posting caption only.", moment.type)
    return None


def _ctx(moment: Moment) -> dict:
    return {
        "headline": moment.headline,
        "blurb": moment.blurb,
        "week": moment.week,
        "period": moment.period_label or f"Week {moment.week}",
        "league": settings.content_league_name,
        "accent": _ACCENT.get(moment.type, "#ff5d3b"),
        "type_label": _LABEL.get(moment.type, moment.type.value.upper()),
        "team_a": moment.team_a, "score_a": moment.score_a,
        "team_b": moment.team_b, "score_b": moment.score_b,
        "big_stat": moment.big_stat, "player": moment.player, "player_team": moment.player_team,
        "lines": moment.lines,
    }


def _render_playwright(moment: Moment, out: Path) -> bool:
    """High-fidelity HTML→PNG. Needs the `write` extra + `playwright install chromium`."""
    try:
        from jinja2 import Template
        from playwright.sync_api import sync_playwright
    except Exception:  # noqa: BLE001
        return False
    try:
        tpl = (Path(__file__).parent / "templates" / "card.html").read_text()
        html = Template(tpl).render(**_ctx(moment))
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(viewport={"width": 1080, "height": 1080})
            page.set_content(html, wait_until="load")
            page.screenshot(path=str(out), clip={"x": 0, "y": 0, "width": 1080, "height": 1080})
            browser.close()
        return out.exists()
    except Exception as e:  # noqa: BLE001  (e.g. browser not installed, or sync API in a loop)
        log.info("Playwright render unavailable (%s); trying Pillow.", e)
        return False


# Candidate bold/regular sans-serif faces across macOS + Linux.
_FONT_BOLD = ["/System/Library/Fonts/Supplemental/Arial Bold.ttf",
              "/System/Library/Fonts/SFNSDisplay-Bold.otf",
              "/Library/Fonts/Arial Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
_FONT_REG = ["/System/Library/Fonts/Supplemental/Arial.ttf",
             "/Library/Fonts/Arial.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]


def _font(size: int, bold: bool):
    from PIL import ImageFont

    for path in (_FONT_BOLD if bold else _FONT_REG):
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:  # noqa: BLE001
                continue
    return ImageFont.load_default()


# Drop glyphs the basic fallback font can't render (emoji, symbols) so we never
# draw "tofu" boxes. Keeps latin + common punctuation/dashes/quotes.
_GLYPH_OK = re.compile(r"[^\x20-ɏ‐-―‘-‟…]")


def _plain(s) -> str:
    return _GLYPH_OK.sub("", str(s or "")).strip() or "?"


def _wrap_px(draw, text: str, font, max_w: float) -> list[str]:
    """Greedy word-wrap by measured pixel width."""
    lines: list[str] = []
    cur = ""
    for word in _plain(text).split():
        trial = f"{cur} {word}".strip()
        if not cur or draw.textlength(trial, font=font) <= max_w:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def _fit_lines(draw, text: str, max_w: float, sizes: list[int], max_lines: int):
    """Pick the largest font size whose wrap fits in max_lines × max_w."""
    for size in sizes:
        font = _font(size, True)
        lines = _wrap_px(draw, text, font, max_w)
        if len(lines) <= max_lines:
            return font, lines, size
    font = _font(sizes[-1], True)
    return font, _wrap_px(draw, text, font, max_w)[:max_lines], sizes[-1]


def _render_pillow(moment: Moment, out: Path) -> bool:
    """Dependency-light fallback card. Always available (Pillow is a core dep)."""
    try:
        from PIL import Image, ImageDraw
    except Exception as e:  # noqa: BLE001
        log.warning("Pillow unavailable (%s).", e)
        return False
    try:
        W = H = 1080
        M = 80                      # margin
        maxw = W - 2 * M            # usable text width
        white, grey, dim = (255, 255, 255), (231, 236, 246), (199, 206, 221)
        accent = _ACCENT.get(moment.type, "#ff5d3b")
        img = Image.new("RGB", (W, H), (12, 15, 26))
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, W, 14], fill=accent)  # accent header bar

        eyebrow = (moment.period_label or f"WEEK {moment.week}").upper()
        wk = eyebrow + (f"  ·  {settings.content_league_name}"
                        if settings.content_league_name else "")
        d.text((M, 70), _plain(wk), font=_font(34, True), fill=grey)
        d.text((M, 120), _plain(_LABEL.get(moment.type, moment.type.value.upper())),
               font=_font(58, True), fill=accent)

        # Headline: largest size that fits in 3 lines.
        hfont, hlines, hsize = _fit_lines(d, moment.headline, maxw, [76, 70, 64, 58, 52], 3)
        y = 232
        for line in hlines:
            d.text((M, y), line, font=hfont, fill=white)
            y += int(hsize * 1.18)

        # Scoreboard (matchup) or a single big stat.
        if moment.score_b is not None:
            y = 600
            for nm, sc in ((moment.team_a, moment.score_a), (moment.team_b, moment.score_b)):
                score = f"{sc:.1f}"
                sfont = _font(88, True)
                sw = d.textlength(score, font=sfont)
                d.text((W - M - sw, y), score, font=sfont, fill=accent)
                nfont, nlines, _ = _fit_lines(d, str(nm), maxw - sw - 60, [46, 40, 34], 1)
                d.text((M, y + 20), nlines[0] if nlines else _plain(nm), font=nfont, fill=grey)
                y += 150
        elif moment.big_stat:
            bfont, blines, _ = _fit_lines(d, moment.big_stat, maxw, [150, 130, 110, 90], 1)
            d.text((M, 600), blines[0] if blines else _plain(moment.big_stat),
                   font=bfont, fill=accent)
            if moment.player:
                pf, pl, _ = _fit_lines(d, moment.player, maxw, [52, 46, 40], 1)
                d.text((M, 770), pl[0] if pl else _plain(moment.player), font=pf, fill=grey)
        elif moment.lines:
            y = 560
            accent_rgb = tuple(int(accent.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
            for ln in moment.lines:
                lf, ll, lsz = _fit_lines(d, ln, maxw - 28, [50, 44, 38], 2)
                d.rectangle([M, y, M + 8, y + len(ll) * int(lsz * 1.2)], fill=accent_rgb)
                for sub in ll:
                    d.text((M + 28, y), sub, font=lf, fill=grey)
                    y += int(lsz * 1.2)
                y += 28

        # Blurb pinned to the bottom (skipped when a lines block already says it).
        if moment.blurb and not moment.lines:
            bl_font = _font(32, False)
            blurb_lines = _wrap_px(d, moment.blurb, bl_font, maxw)[:3]
            y = H - M - len(blurb_lines) * 46
            for line in blurb_lines:
                d.text((M, y), line, font=bl_font, fill=dim)
                y += 46

        img.save(out, "PNG")
        return out.exists()
    except Exception as e:  # noqa: BLE001
        log.warning("Pillow render failed (%s).", e)
        return False
