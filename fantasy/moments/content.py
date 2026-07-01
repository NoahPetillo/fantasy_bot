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

from fantasy.moments.config import content_config as settings  # decoupled from the app
from fantasy.moments.models import Moment, MomentType

log = logging.getLogger(__name__)

ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"  # matches fantasy/news/extract.py
XAI_MODEL = "grok-4.3"                          # xAI's recommended fast model
_XAI_URL = "https://api.x.ai/v1/chat/completions"

_ACCENT = {
    MomentType.matchup: "#3a86ff",
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
    MomentType.matchup: "🏈",
    MomentType.nailbiter: "😱", MomentType.blowout: "💀", MomentType.high_score: "👑",
    MomentType.low_score: "🚮", MomentType.bench_blunder: "🪑", MomentType.lucky: "🍀",
    MomentType.unlucky: "🥲", MomentType.boom: "🚀", MomentType.bust: "🥚",
    MomentType.hot_streak: "🔥", MomentType.cold_streak: "🥶", MomentType.rivalry: "😤",
    MomentType.trade: "🤝", MomentType.waiver: "💰",
}
_LABEL = {
    MomentType.matchup: "RESULT",
    MomentType.nailbiter: "NAIL-BITER", MomentType.blowout: "BLOWOUT",
    MomentType.high_score: "TOP SCORE", MomentType.low_score: "LOW SCORE",
    MomentType.bench_blunder: "BENCH BLUNDER", MomentType.lucky: "LUCKY W",
    MomentType.unlucky: "UNLUCKY L", MomentType.boom: "BOOM", MomentType.bust: "BUST",
    MomentType.hot_streak: "HOT STREAK", MomentType.cold_streak: "COLD STREAK",
    MomentType.rivalry: "RIVALRY", MomentType.trade: "TRADE ALERT", MomentType.waiver: "FAAB SPLASH",
}

# Disgust vocabulary, rotated per moment so captions don't all reach for the same
# word (the LLM anchors hard on whatever example it's handed).
_DISGUST_WORDS = ["vile", "putrid", "awful", "disgusting", "atrocious", "revolting",
                  "gross", "abysmal", "repugnant", "rancid", "wretched", "horrid",
                  "grotesque", "sorry", "sad", "laughable", "poverty"]


def _flavor_words(moment: Moment, n: int = 4) -> list[str]:
    """A rotating slice of disgust adjectives, deterministic per moment, so each
    caption is nudged toward a different one instead of leaning on 'vile'."""
    h = int(hashlib.sha1(
        f"{moment.dedup_key}:{moment.type.value}:{moment.week}".encode()).hexdigest(), 16)
    start = h % len(_DISGUST_WORDS)
    return [_DISGUST_WORDS[(start + i) % len(_DISGUST_WORDS)] for i in range(n)]


# ── caption ──────────────────────────────────────────────────────────────────
def write_caption(moment: Moment) -> str:
    """A short, punchy caption. LLM if a provider key is configured, else a template."""
    return _llm_caption(_caption_prompt(moment)) or _fallback_caption(moment)


def _resolve_provider() -> tuple[str | None, str | None, str | None]:
    """(provider, model, api_key) per config; (None, None, None) if no key available.

    ``auto`` prefers Groq (generous free tier — what the rest of the app prefers),
    then xAI/Grok, then Anthropic. ``content_llm_model`` overrides any default.
    """
    prov = (settings.content_llm_provider or "auto").strip().lower()
    if prov == "grok":
        prov = "xai"
    override = settings.content_llm_model
    table = {  # provider -> (api_key, default_model)
        "groq": (settings.groq_api_key, override or settings.groq_model),
        "xai": (settings.xai_api_key, override or XAI_MODEL),
        "anthropic": (settings.anthropic_api_key, override or ANTHROPIC_MODEL),
    }
    if prov in table:
        key, model = table[prov]
        return (prov, model, key) if key else (None, None, None)
    if prov == "auto":
        for p in ("groq", "xai", "anthropic"):
            key, model = table[p]
            if key:
                return p, model, key
    return None, None, None


def _llm_caption(prompt: str) -> str | None:
    provider, model, key = _resolve_provider()
    if not provider:
        return None
    try:
        if provider == "groq":
            return _groq_complete(prompt, model, key)
        if provider == "xai":
            return _xai_complete(prompt, model, key)
        return _anthropic_complete(prompt, model, key)
    except Exception as e:  # noqa: BLE001
        log.warning("Caption LLM (%s) failed (%s); falling back to template.", provider, e)
        return None


def _groq_complete(prompt: str, model: str, key: str) -> str | None:
    from groq import Groq  # official SDK (same path the chat agent uses)

    resp = Groq(api_key=key).chat.completions.create(
        model=model, max_tokens=300, temperature=1.0,
        messages=[{"role": "user", "content": prompt}],
    )
    return (resp.choices[0].message.content or "").strip() or None


def _anthropic_complete(prompt: str, model: str, key: str) -> str | None:
    from anthropic import Anthropic

    resp = Anthropic(api_key=key).messages.create(
        model=model, max_tokens=200, messages=[{"role": "user", "content": prompt}]
    )
    return "".join(
        getattr(b, "text", "") for b in resp.content if getattr(b, "type", None) == "text"
    ).strip() or None


def _xai_complete(prompt: str, model: str, key: str) -> str | None:
    import requests

    resp = requests.post(
        _XAI_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": model, "max_tokens": 200, "temperature": 1.0,
              "messages": [{"role": "user", "content": prompt}]},
        timeout=30,
    )
    resp.raise_for_status()
    choices = (resp.json() or {}).get("choices") or []
    return (choices[0]["message"]["content"].strip() if choices else None) or None


def _caption_prompt(moment: Moment) -> str:
    from fantasy.moments.roasts import roast_for

    league = settings.content_league_name or "the league"
    style = (settings.content_style_note or "").strip()
    roast = roast_for(moment.manager, moment.week)
    who = f"\nCALL THIS PERSON OUT BY NAME: {moment.manager}" if moment.manager else ""
    if settings.content_voice == "instagram":
        voice = (
            "Write a cocky, punchy Instagram caption (1–2 sentences) with playful trash talk, then "
            "3–5 hashtags on a new line. Keep it edgy but public-friendly — no hard profanity.")
    else:  # group_chat = savage (default)
        voice = (
            "Write 1–2 sentences of savage, sarcastic group-chat trash talk in the voice of a tight "
            "friend group that mercilessly busts each other's balls over fantasy football. Be crude, "
            "profane and specific — swearing (shit/fuck/etc.) is encouraged. Channel a line like "
            "\"that start/sit was laughable, fuck you {name}\". When you reach for a disgust word, "
            "VARY it — pull from options like " + ", ".join(_flavor_words(moment)) + " — and don't "
            "lean on the same adjective every time. No hashtags, no corny hype-bot energy, and don't "
            "explain the joke.")
    guard = (
        "This is affectionate ball-busting between friends — roast their garbage fantasy decisions "
        "as hard as you want, but NO slurs and no bigotry; never attack anyone's race, gender, "
        "religion, or the like. Keep it about their terrible team.")
    extra = f"\nOVERALL GROUP VIBE: {style}" if style else ""
    joke = (f"\nINSIDE JOKE TO WORK IN (land it naturally, don't force it): {roast}"
            if roast else "")
    return (
        f"You are the resident shit-talker for {league}. {voice}\n{guard}{extra}{joke}\n"
        f"Use ONLY these facts — keep all names and numbers exact, invent nothing:\n"
        f"FACTS: {moment.blurb}{who}\n"
        f"Return only the caption text, nothing else."
    )


# Savage templated kickers for when no LLM key is configured — edgier than a bare
# restatement, but the LLM is where the real venom lives.
def _savage_kicker(moment: Moment) -> str:
    n = moment.manager
    callout = f"fuck you, {n}" if n else "just laughable"
    named = f"{n}, " if n else ""
    w = _flavor_words(moment)[0]  # rotating disgust word, not always "vile"
    return {
        MomentType.nailbiter: f"{named}you nearly shit the bed on that one.",
        MomentType.blowout: f"that wasn't a game, it was a public execution. {named}embarrassing.",
        MomentType.high_score: "alright, flex on the poors, we get it.",
        MomentType.low_score: f"genuinely {w}. {named}that lineup belongs in a dumpster.",
        MomentType.bench_blunder: f"a clinic in coaching malpractice — {callout}.",
        MomentType.lucky: f"backed into a W like the absolute fraud {n or 'they are'}.",
        MomentType.unlucky: f"dropped a ton and still lost. {named}the schedule has it out for you, lmao.",
        MomentType.boom: "went nuclear. cute pickup you'll fumble away next week.",
        MomentType.bust: f"laid a colossal egg — {callout}.",
        MomentType.hot_streak: "on a heater. enjoy it before the inevitable collapse.",
        MomentType.cold_streak: f"in total free fall. {named}it's getting hard to watch.",
        MomentType.rivalry: "rivalry settled — somebody go talk their shit.",
        MomentType.trade: "a trade?! in this economy of cowards? somebody finally grew a pair.",
        MomentType.waiver: f"dropped real money on a waiver. {named}better not be a damn kicker.",
    }.get(moment.type, "woof.")


def _fallback_caption(moment: Moment) -> str:
    from fantasy.moments.roasts import roast_for

    emoji = _EMOJI.get(moment.type, "🔥")
    if settings.content_voice == "instagram":
        tags = {
            MomentType.nailbiter: "#nailbiter #fantasyfootball #closegame",
            MomentType.blowout: "#blowout #fantasyfootball #domination",
            MomentType.bench_blunder: "#benchwoes #fantasyfootball #coachingmalpractice",
            MomentType.unlucky: "#unlucky #fantasyfootball #robbed",
            MomentType.lucky: "#luckywin #fantasyfootball #scheduleluck",
            MomentType.boom: "#boom #fantasyfootball #league",
            MomentType.bust: "#bust #fantasyfootball #benchhim",
            MomentType.hot_streak: "#hotstreak #fantasyfootball #rolling",
            MomentType.cold_streak: "#coldstreak #fantasyfootball #freefall",
            MomentType.rivalry: "#rivalry #fantasyfootball #badblood",
            MomentType.trade: "#trade #fantasyfootball #blockbuster",
            MomentType.waiver: "#waivers #faab #fantasyfootball",
        }.get(moment.type, "#fantasyfootball #league")
        return f"{emoji} {moment.blurb}\n{tags}"
    roast = roast_for(moment.manager, moment.week)
    tail = f" {roast}" if roast else ""
    return f"{emoji} {moment.blurb} {_savage_kicker(moment)}{tail}"


def card_header(moment: Moment) -> str:
    """Short Discord message line posted alongside the image (the funny caption is
    baked INTO the image, so the message itself stays a compact label)."""
    emoji = _EMOJI.get(moment.type, "🔥")
    label = _LABEL.get(moment.type, moment.type.value.upper()).title()
    period = moment.period_label or f"Week {moment.week}"
    return f"{emoji} {label} · {period}"


# ── graphic ──────────────────────────────────────────────────────────────────
def render_card(moment: Moment, caption: str | None = None, out_dir: Path | None = None) -> Path | None:
    """Render a 1080×1080 PNG with the CAPTION baked in as the hero text. Returns
    the path, or None if no renderer is available (caption still posts as text)."""
    caption = caption or moment.blurb
    out_dir = out_dir or settings.media_dir
    h = hashlib.sha1(f"{moment.type.value}:{moment.dedup_key}".encode()).hexdigest()[:8]
    out = Path(out_dir) / f"wk{moment.week:02d}_{moment.type.value}_{h}.png"
    if _render_playwright(moment, out, caption):
        return out
    if _render_pillow(moment, out, caption):
        return out
    log.warning("No graphic renderer available for moment %s; posting caption only.", moment.type)
    return None


def _ctx(moment: Moment, caption: str) -> dict:
    return {
        "caption": caption,
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


def _render_playwright(moment: Moment, out: Path, caption: str) -> bool:
    """High-fidelity HTML→PNG. Needs the `write` extra + `playwright install chromium`."""
    try:
        from jinja2 import Template
        from playwright.sync_api import sync_playwright
    except Exception:  # noqa: BLE001
        return False
    try:
        tpl = (Path(__file__).parent / "templates" / "card.html").read_text()
        html = Template(tpl).render(**_ctx(moment, caption))
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


def _fit_block(draw, text: str, max_w: float, max_h: float, sizes: list[int]):
    """Largest bold size whose wrapped text fits within max_w × max_h."""
    for size in sizes:
        font = _font(size, True)
        lines = _wrap_px(draw, text, font, max_w)
        lh = int(size * 1.22)
        if len(lines) * lh <= max_h:
            return font, lines, lh
    size = sizes[-1]
    font = _font(size, True)
    lh = int(size * 1.22)
    lines = _wrap_px(draw, text, font, max_w)
    return font, lines[: max(1, int(max_h // lh))], lh


def _render_pillow(moment: Moment, out: Path, caption: str) -> bool:
    """Dependency-light fallback card, with the caption baked in as the hero text.
    Always available (Pillow is a core dep)."""
    try:
        from PIL import Image, ImageDraw
    except Exception as e:  # noqa: BLE001
        log.warning("Pillow unavailable (%s).", e)
        return False
    try:
        W = H = 1080
        M = 80                      # margin
        maxw = W - 2 * M            # usable text width
        white, grey = (255, 255, 255), (231, 236, 246)
        accent = _ACCENT.get(moment.type, "#ff5d3b")
        accent_rgb = tuple(int(accent.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
        img = Image.new("RGB", (W, H), (12, 15, 26))
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, W, 14], fill=accent)  # accent header bar

        eyebrow = (moment.period_label or f"WEEK {moment.week}").upper()
        wk = eyebrow + (f"  ·  {settings.content_league_name}"
                        if settings.content_league_name else "")
        d.text((M, 66), _plain(wk), font=_font(32, True), fill=grey)
        d.text((M, 110), _plain(_LABEL.get(moment.type, moment.type.value.upper())),
               font=_font(52, True), fill=accent)

        # ── numbers zone (compact, upper) — the factual anchor ──
        y = 208
        if moment.score_b is not None:
            for nm, sc in ((moment.team_a, moment.score_a), (moment.team_b, moment.score_b)):
                score = f"{sc:.1f}"
                sfont = _font(58, True)
                sw = d.textlength(score, font=sfont)
                d.text((W - M - sw, y), score, font=sfont, fill=accent)
                nfont, nlines, _ = _fit_lines(d, str(nm), maxw - sw - 40, [42, 36, 30], 1)
                d.text((M, y + 12), nlines[0] if nlines else _plain(nm), font=nfont, fill=grey)
                y += 88
        elif moment.big_stat:
            bfont, blines, bsz = _fit_lines(d, moment.big_stat, maxw, [120, 104, 88], 1)
            d.text((M, y), blines[0] if blines else _plain(moment.big_stat), font=bfont, fill=accent)
            y += int(bsz * 1.12)
            if moment.player:
                pf, pl, _ = _fit_lines(d, moment.player, maxw, [46, 40, 34], 1)
                d.text((M, y), pl[0] if pl else _plain(moment.player), font=pf, fill=grey)
                y += 58
        elif moment.lines:
            for ln in moment.lines:
                lf, ll, lsz = _fit_lines(d, ln, maxw - 28, [44, 38, 34], 2)
                d.rectangle([M, y, M + 8, y + len(ll) * int(lsz * 1.2)], fill=accent_rgb)
                for sub in ll:
                    d.text((M + 28, y), sub, font=lf, fill=grey)
                    y += int(lsz * 1.2)
                y += 20

        # ── caption hero (the funny line, fills the rest) ──
        cap_top = max(y, 380) + 46
        d.rectangle([M, cap_top - 28, M + 96, cap_top - 21], fill=accent_rgb)  # divider tick
        cfont, clines, lh = _fit_block(d, caption, maxw, H - M - cap_top,
                                       [60, 54, 48, 44, 40, 36, 32])
        yy = cap_top
        for line in clines:
            d.text((M, yy), line, font=cfont, fill=white)
            yy += lh

        img.save(out, "PNG")
        return out.exists()
    except Exception as e:  # noqa: BLE001
        log.warning("Pillow render failed (%s).", e)
        return False
