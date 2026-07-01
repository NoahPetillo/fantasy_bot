"""Central configuration.

Loads runtime settings from environment / ``.env`` (secrets, IDs, mode) and
exposes a single ``settings`` object the rest of the app imports.

League *scoring/roster* settings are NOT hardcoded here — they are read live
from ESPN's ``mSettings`` (see :mod:`fantasy.espn.client`) so every valuation is
parameterized by the user's actual league.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ExecutionMode(str, Enum):
    """How far the system is allowed to act on its own decisions.

    - ``advise``  : produce recommendations only; never write anywhere (Phases 0-2).
    - ``approve`` : push proposals for human Approve/Reject, execute on approval (Phase 3).
    - ``auto``    : may auto-execute ONLY a strictly-better lineup before lock;
                    never auto-executes waivers or trades (those always need approval).
    """

    advise = "advise"
    approve = "approve"
    auto = "auto"


class ExecutionBackend(str, Enum):
    """How an approved action is carried out.

    - ``deeplink``   : generate a one-tap ESPN URL + instructions (no write; safest).
    - ``playwright`` : drive the ESPN web UI to perform the write (needs cookies).
    - ``dryrun``     : simulate the write (for testing the approval→execute flow).
    """

    deeplink = "deeplink"
    playwright = "playwright"
    dryrun = "dryrun"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # .env.local is where `clerk env pull` writes its keys; it takes precedence.
        env_file=(".env", ".env.local"), env_file_encoding="utf-8",
        extra="ignore", case_sensitive=False,
    )

    # ── ESPN ──
    espn_s2: str | None = Field(default=None, alias="ESPN_S2")
    espn_swid: str | None = Field(default=None, alias="ESPN_SWID")
    espn_league_id: int | None = Field(default=None, alias="ESPN_LEAGUE_ID")
    espn_season: int = Field(default=2025, alias="ESPN_SEASON")
    espn_team_id: int | None = Field(default=None, alias="ESPN_TEAM_ID")

    # ── LLM ──
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    # Groq (NOTE: Groq the inference provider, key prefix `gsk_` — NOT xAI's Grok
    # below). Fast OpenAI-compatible inference for open models; powers the chatbot's
    # free-form Q&A when set (preferred over Anthropic, then the keyless parser).
    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.3-70b-versatile", alias="GROQ_MODEL")
    # xAI / Grok — OpenAI-compatible alternative caption provider. Accepts either
    # XAI_API_KEY or GROK_API_KEY in the environment.
    xai_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("XAI_API_KEY", "GROK_API_KEY")
    )

    # ── Player props (the-odds-api.com; optional, the sharpest projection source) ──
    odds_api_key: str | None = Field(default=None, alias="ODDS_API_KEY")

    # ── Expert signals: X/Twitter (OFF by default; free Bluesky+RSS is the default) ──
    enable_x_source: bool = Field(default=False, alias="ENABLE_X_SOURCE")
    x_bearer_token: str | None = Field(default=None, alias="X_BEARER_TOKEN")
    # Ingest expert/news signals at all (network). Decision-adjust applies the
    # gated, capped deltas/boosts to projections + waivers; off => alert-only.
    enable_expert_signals: bool = Field(default=True, alias="ENABLE_EXPERT_SIGNALS")
    expert_adjust_decisions: bool = Field(default=True, alias="EXPERT_ADJUST_DECISIONS")

    # ── Notifications ──
    slack_bot_token: str | None = Field(default=None, alias="SLACK_BOT_TOKEN")
    slack_app_token: str | None = Field(default=None, alias="SLACK_APP_TOKEN")
    slack_channel_id: str | None = Field(default=None, alias="SLACK_CHANNEL_ID")
    ntfy_topic: str | None = Field(default=None, alias="NTFY_TOPIC")

    # ── Content engine (league hype posts) ──
    # Where approved moments get posted. A Discord channel webhook URL is the
    # whole setup — no bot, no OAuth. (Instagram is deliberately NOT automated:
    # the engine produces a ready-to-post image you upload to IG by hand.)
    discord_webhook_url: str | None = Field(default=None, alias="DISCORD_WEBHOOK_URL")
    # Auto-post (skip human approval) — the hands-off deploy mode. When on, any
    # generated moment with spice >= content_autopost_min_spice posts straight to
    # Discord and is marked executed. Off by default (approve-first).
    content_autopost: bool = Field(default=False, alias="CONTENT_AUTOPOST")
    content_autopost_min_spice: float = Field(default=0.0, alias="CONTENT_AUTOPOST_MIN_SPICE")
    content_moments_per_week: int = Field(default=3, alias="CONTENT_MOMENTS_PER_WEEK")
    content_nailbiter_margin: float = Field(default=5.0, alias="CONTENT_NAILBITER_MARGIN")
    content_blowout_margin: float = Field(default=40.0, alias="CONTENT_BLOWOUT_MARGIN")
    content_bench_blunder_min: float = Field(default=8.0, alias="CONTENT_BENCH_BLUNDER_MIN")
    # Caption LLM provider: "auto" (use whichever key is set — prefers Groq, then
    # xAI/Grok, then Anthropic), or force "groq" | "xai"/"grok" | "anthropic".
    # content_llm_model overrides the per-provider default (groq -> GROQ_MODEL,
    # xai -> grok-4.3, anthropic -> claude-haiku-4-5).
    content_llm_provider: str = Field(default="auto", alias="CONTENT_LLM_PROVIDER")
    content_llm_model: str | None = Field(default=None, alias="CONTENT_LLM_MODEL")
    # Caption tone: "group_chat" (savage, profane friend-group trash talk — the
    # default) or "instagram" (cocky but public-safe, +hashtags).
    content_voice: str = Field(default="group_chat", alias="CONTENT_VOICE")
    content_league_name: str | None = Field(default=None, alias="CONTENT_LEAGUE_NAME")
    # Optional one-line ALWAYS-ON vibe note woven into every caption prompt
    # (e.g. "extra mean, lots of swearing"). Per-person inside jokes go in the
    # roast book (config/roasts.yaml), not here.
    content_style_note: str | None = Field(default=None, alias="CONTENT_STYLE_NOTE")
    # Roast book: per-league inside jokes sprinkled in occasionally.
    content_roasts_file: Path = Field(default=Path("config/roasts.yaml"),
                                      alias="CONTENT_ROASTS_FILE")
    # A manager's inside joke fires roughly 1-in-N of the moments they star in
    # (3 ≈ "every few weeks"). Lower = more often, higher = rarer.
    content_roast_frequency: int = Field(default=3, alias="CONTENT_ROAST_FREQUENCY")
    # Phase 2 moment tuning.
    content_streak_min: int = Field(default=3, alias="CONTENT_STREAK_MIN")  # min W/L run to flag
    content_min_faab_bid: int = Field(default=15, alias="CONTENT_MIN_FAAB_BID")  # min $ to flag
    # Rivalry pairs — list of [tokenA, tokenB], each token a team name/abbrev/id.
    # e.g. CONTENT_RIVALRIES='[["Maye shots","Emeka the Freaka"],["6","9"]]'
    content_rivalries: list[list[str]] | None = Field(default=None, alias="CONTENT_RIVALRIES")

    # ── Chatbot abuse floor ──
    # Chat is a logged-in (Clerk) feature; this per-IP cap is an abuse floor on top
    # of auth (a per-user plan quota lands in Phase 5). Set CHAT_RATE_LIMIT=0 to
    # disable. Generous by default.
    chat_rate_limit: int = Field(default=250, alias="CHAT_RATE_LIMIT")
    chat_rate_window_seconds: int = Field(default=3600, alias="CHAT_RATE_WINDOW_SECONDS")

    # ── Runtime ──
    execution_mode: ExecutionMode = Field(default=ExecutionMode.advise, alias="EXECUTION_MODE")
    execution_backend: ExecutionBackend = Field(
        default=ExecutionBackend.deeplink, alias="EXECUTION_BACKEND"
    )
    # Multi-source consensus projection (model + ESPN + Sleeper avg) vs ESPN-primary.
    projection_consensus: bool = Field(default=True, alias="PROJECTION_CONSENSUS")
    # Surface trade proposals FIRST in notifications + flag them — the decision
    # audit shows the trade market is this manager's single biggest edge.
    prioritize_trades: bool = Field(default=True, alias="PRIORITIZE_TRADES")
    data_dir: Path = Field(default=Path("./data"), alias="DATA_DIR")

    # ── Multi-tenant SaaS infrastructure (see MULTITENANT_BUILD.md) ──
    # Postgres (Neon) connection string. When unset, the app falls back to a
    # local SQLite file so dev/CI boot without a database — see
    # ``effective_database_url``. Production MUST set this to the Neon URL.
    database_url: str | None = Field(default=None, alias="DATABASE_URL")
    # Fernet key that encrypts users' ESPN cookies at rest. Held in the host's
    # secret store, NEVER in the database. Generate with
    # ``python -m fantasy.security.crypto``. Required before any ESPN cookie is
    # stored (Phase 2); optional in Phase 1 so the app still boots without it.
    credential_enc_key: str | None = Field(default=None, alias="CREDENTIAL_ENC_KEY")
    # Clerk (managed auth). The backend verifies the Clerk session JWT via JWKS on
    # each request; the secret key is only needed for server-side Clerk API calls
    # (e.g. email backfill). Set ``clerk_issuer`` (or ``clerk_jwks_url``) so the
    # JWKS endpoint comes from trusted config — it is NEVER derived from the token
    # itself (that would allow issuer spoofing). Auth fails closed if unset.
    # Accept whichever variable name the Clerk CLI / host writes.
    clerk_secret_key: str | None = Field(
        default=None, validation_alias=AliasChoices("CLERK_SECRET_KEY", "CLERK_API_KEY"))
    clerk_publishable_key: str | None = Field(
        default=None, validation_alias=AliasChoices(
            "CLERK_PUBLISHABLE_KEY", "NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY",
            "VITE_CLERK_PUBLISHABLE_KEY", "CLERK_PUBLISHABLE"))
    clerk_jwks_url: str | None = Field(default=None, alias="CLERK_JWKS_URL")
    clerk_issuer: str | None = Field(default=None, alias="CLERK_ISSUER")
    # Product name shown in the consent screen / legal copy (fills [PRODUCT_NAME]).
    product_name: str = Field(default="Fantasy Copilot", alias="PRODUCT_NAME")
    # Stripe (Phase 5 — subscriptions). Secret key for API calls, webhook secret to
    # verify events, price id of the Pro plan. Unset → billing disabled (free only).
    stripe_secret_key: str | None = Field(default=None, alias="STRIPE_SECRET_KEY")
    stripe_webhook_secret: str | None = Field(default=None, alias="STRIPE_WEBHOOK_SECRET")
    stripe_price_id: str | None = Field(default=None, alias="STRIPE_PRICE_ID")

    @property
    def espn_swid_braced(self) -> str | None:
        """SWID with surrounding braces, which ESPN expects."""
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
    def effective_database_url(self) -> str:
        """The SQLAlchemy URL to use. Prefers ``DATABASE_URL`` (Neon in prod);
        falls back to a local SQLite file so dev/CI boot without a database.

        Also normalizes the legacy ``postgres://`` scheme (what some hosts inject)
        to the ``postgresql+psycopg://`` driver URL SQLAlchemy 2.0 expects.
        """
        url = self.database_url
        if not url:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            return f"sqlite:///{(self.data_dir / 'app.sqlite').as_posix()}"
        if url.startswith("postgres://"):
            url = "postgresql+psycopg://" + url[len("postgres://"):]
        elif url.startswith("postgresql://"):
            url = "postgresql+psycopg://" + url[len("postgresql://"):]
        return url

    @property
    def cache_dir(self) -> Path:
        d = self.data_dir / "cache"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def db_path(self) -> Path:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self.data_dir / "fantasy.sqlite"

    @property
    def media_dir(self) -> Path:
        """Where generated moment graphics are written (and read for posting)."""
        d = self.data_dir / "media"
        d.mkdir(parents=True, exist_ok=True)
        return d


settings = Settings()
