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

from pydantic import Field
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
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    # ── ESPN ──
    espn_s2: str | None = Field(default=None, alias="ESPN_S2")
    espn_swid: str | None = Field(default=None, alias="ESPN_SWID")
    espn_league_id: int | None = Field(default=None, alias="ESPN_LEAGUE_ID")
    espn_season: int = Field(default=2025, alias="ESPN_SEASON")
    espn_team_id: int | None = Field(default=None, alias="ESPN_TEAM_ID")

    # ── LLM ──
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")

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
    def cache_dir(self) -> Path:
        d = self.data_dir / "cache"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def db_path(self) -> Path:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self.data_dir / "fantasy.sqlite"


settings = Settings()
