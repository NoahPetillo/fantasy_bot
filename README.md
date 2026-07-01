# Fantasy App — Autonomous ESPN Fantasy Football AI

A system that plays fantasy football at a high level on an **ESPN** league: a
league-adaptive player-valuation + projection model drives every recurring
decision (draft, start/sit, waivers/FAAB, trades), it reads real-world NFL news
to stay current, runs on continuous loops, and proactively **notifies** you of
high-value, likely-to-be-accepted moves. The model is validated by backtesting
against past seasons and improved via draft self-play.

**Posture:** notify-and-approve. The bot decides everything and pushes a rich
notification; you tap Approve/Reject. The entire "brain" runs on read-only ESPN
data plus public NFL data — no risky unattended writes.

## League-adaptive by design

Nothing is hardcoded to PPR or 12 teams. The league's full configuration —
scoring rules (the complete stat→points map), roster slots (FLEX / superflex /
TE-premium / IDP / K / DST / bench / IR), team count, waiver type + FAAB budget,
playoff weeks, keeper/dynasty flags — is read live from ESPN `mSettings` into a
single first-class `LeagueSettings` object (`fantasy/league_settings.py`) that
every valuation reads from. The same `ScoringEngine` scores both projections and
historical backtests using *your* rules.

## Status

Greenfield, in active build. See the plan at
`~/.claude/plans/i-want-to-create-snappy-dream.md`.

- **Phase 0** — prove ESPN read access on the real league. *(scaffold done; needs cookies)*
- **Phase 1** — ID crosswalk + projection model + VOR + backtest (beat ESPN's projections).
- **Phase 2** — notify-only loop (FastAPI + scheduler + Slack), news ingestion.
- **Phase 3** — write execution behind the approval gate (lineup → waivers → trades).
- **Phase 4** — live draft assistant + self-play.
- **Phase 5** — polished unified dashboard.

## Setup

```bash
# uv manages a Python 3.12 venv (3.14 is too new for the ML wheels)
uv sync --extra dev

# configure
cp .env.example .env   # fill in ESPN cookies + league id

# Phase 0 gate
uv run python scripts/verify_espn_read.py

# tests
uv run pytest -q
```

## Layout

```
fantasy/
  config.py            # runtime settings (.env)
  league_settings.py   # LeagueSettings — the league-adaptive backbone
  espn/                # read client, stat-id maps, (later) write tier
  data/                # nflverse loaders, id crosswalk, ADP, market values
  projections/         # per-position models, distributions
  valuation/           # scoring engine, VOR engine
  decisions/           # draft, start/sit, FAAB, lineup LP, trades
  news/                # ingesters + LLM signal extraction
  backtest/            # season sim, draft sim, self-play
  orchestrator/        # scheduler, loops, action log, idempotency
  notify/              # Slack / ntfy
  api/                 # FastAPI: approval webhook, draft assistant, dashboard
  moments/             # league content engine (savage recap cards → Discord)
```

## League Content Bot (savage weekly recaps → Discord)

`scripts/content_bot.py` scans your league's most recently finished week, makes a
card + trash-talk caption for **every matchup** plus a handful of extra "moments"
(worst bench decision, lowest score, boom/bust, win streaks…), and posts them to
your league's Discord. The funny caption is baked onto the image itself.

### Run it

Always run from the **project root** (the `fantasy_app/` folder that has
`pyproject.toml`):

```bash
cd /Users/noahpetillo/Projects/fantasy_app

echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# Tuesday morning: preview the week, then confirm to post
uv run python scripts/content_bot.py

# Preview only — generate + print, never posts
uv run python scripts/content_bot.py --dry-run

# Post everything with no prompt (e.g. from a cron job)
uv run python scripts/content_bot.py --yes

# Recap a specific week
uv run python scripts/content_bot.py --week 14

# Run continuously (posts Tue 10am ET + trades/waivers every 6h)
uv run python scripts/content_bot.py --schedule
```

A normal run previews the cards (captions + image paths) and asks
`Post all N to Discord? [y/N]` — nothing hits the chat until you say yes. Each
moment posts at most once (tracked in `data/content_bot.sqlite`), so previewing
or cancelling never "uses it up".

### Setup — `.env` in the project root

```
ESPN_S2=...              # ESPN cookies (private-league access)
ESPN_SWID=...
ESPN_LEAGUE_ID=...
ESPN_SEASON=2025
ESPN_TEAM_ID=...
DISCORD_WEBHOOK_URL=...  # the channel it posts to (Server Settings → Integrations → Webhooks)
GROQ_API_KEY=gsk_...     # writes the captions (free tier — console.groq.com)
CONTENT_LEAGUE_NAME=...  # optional, printed on the cards
```

Optional tuning (also `.env`):

```
CONTENT_EXTRA_MOMENTS=5   # extra superlatives on top of the per-matchup recaps
CONTENT_VOICE=group_chat  # savage/profane (default) | instagram (public-safe)
CONTENT_RIVALRIES=[["Team A","Team B"]]   # flag rivalry games
CONTENT_ROAST_FREQUENCY=3 # inside jokes (config/roasts.yaml) fire ~1-in-N weeks
GROQ_MODEL=llama-3.3-70b-versatile        # swap if it gets deprecated
```

Per-person inside jokes live in `config/roasts.yaml` (keyed by league + ESPN first name).

