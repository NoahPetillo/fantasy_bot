"""Content-engine configuration — self-contained and decoupled from the app.

The rest of the codebase is migrating to multi-tenant (per-user ESPN cookies in
Postgres, Clerk auth — see MULTITENANT_BUILD.md), which reshapes
``fantasy/config.py``, the ``Store``, and how ``EspnClient`` gets its cookies. The
content engine is a personal, single-tenant feature, so it reads its config from
THIS object instead of the global ``fantasy.config.settings`` — the SaaS refactor
can gut those without breaking the content bot.

It loads the SAME environment variables the app used, so nothing changes
operationally today. The standalone runner (``scripts/content_bot.py``) drives
everything off this object + a dedicated SQLite store.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ContentConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    # ── ESPN (the bot builds its own read-only client from these) ──
    espn_s2: str | None = Field(default=None, alias="ESPN_S2")
    espn_swid: str | None = Field(default=None, alias="ESPN_SWID")
    espn_league_id: int | None = Field(default=None, alias="ESPN_LEAGUE_ID")
    espn_season: int = Field(default=2025, alias="ESPN_SEASON")
    espn_team_id: int | None = Field(default=None, alias="ESPN_TEAM_ID")

    # ── Caption LLM (providers + keys stay app-level infra) ──
    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.3-70b-versatile", alias="GROQ_MODEL")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    xai_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("XAI_API_KEY", "GROK_API_KEY")
    )
    content_llm_provider: str = Field(default="auto", alias="CONTENT_LLM_PROVIDER")
    content_llm_model: str | None = Field(default=None, alias="CONTENT_LLM_MODEL")

    # ── Posting ──
    discord_webhook_url: str | None = Field(default=None, alias="DISCORD_WEBHOOK_URL")
    content_autopost: bool = Field(default=False, alias="CONTENT_AUTOPOST")
    content_autopost_min_spice: float = Field(default=0.0, alias="CONTENT_AUTOPOST_MIN_SPICE")

    # ── Moment tuning ──
    # Every matchup gets a recap card; this is how many EXTRA superlatives
    # (bench blunders, low/high scores, boom/bust, streaks…) to add on top.
    content_extra_moments: int = Field(default=5, alias="CONTENT_EXTRA_MOMENTS")
    content_moments_per_week: int = Field(default=3, alias="CONTENT_MOMENTS_PER_WEEK")  # legacy
    content_nailbiter_margin: float = Field(default=5.0, alias="CONTENT_NAILBITER_MARGIN")
    content_blowout_margin: float = Field(default=40.0, alias="CONTENT_BLOWOUT_MARGIN")
    content_bench_blunder_min: float = Field(default=8.0, alias="CONTENT_BENCH_BLUNDER_MIN")
    content_streak_min: int = Field(default=3, alias="CONTENT_STREAK_MIN")
    content_min_faab_bid: int = Field(default=15, alias="CONTENT_MIN_FAAB_BID")
    content_rivalries: list[list[str]] | None = Field(default=None, alias="CONTENT_RIVALRIES")

    # ── Voice / roast book ──
    content_voice: str = Field(default="group_chat", alias="CONTENT_VOICE")
    content_league_name: str | None = Field(default=None, alias="CONTENT_LEAGUE_NAME")
    content_style_note: str | None = Field(default=None, alias="CONTENT_STYLE_NOTE")
    content_roasts_file: Path = Field(default=Path("config/roasts.yaml"), alias="CONTENT_ROASTS_FILE")
    content_roast_frequency: int = Field(default=3, alias="CONTENT_ROAST_FREQUENCY")

    # ── Storage ──
    data_dir: Path = Field(default=Path("./data"), alias="DATA_DIR")

    @property
    def espn_swid_braced(self) -> str | None:
        """SWID wrapped in braces, which ESPN expects."""
        if not self.espn_swid:
            return None
        s = self.espn_swid.strip()
        if not s.startswith("{"):
            s = "{" + s
        if not s.endswith("}"):
            s = s + "}"
        return s

    @property
    def has_espn_auth(self) -> bool:
        return bool(self.espn_s2 and self.espn_swid and self.espn_league_id)

    @property
    def media_dir(self) -> Path:
        d = self.data_dir / "media"
        d.mkdir(parents=True, exist_ok=True)
        return d


content_config = ContentConfig()
