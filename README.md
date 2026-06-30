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
```
