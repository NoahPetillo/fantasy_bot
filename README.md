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

All five phases have working implementations; the app is deployed as a
multi-tenant SaaS (Clerk + Postgres) and ready for the 2026 season.

- **Phase 0** — ESPN read access: verified live on the real league.
- **Phase 1** — projection model + VOR + backtest. Current accuracy (train
  2021-24, test 2025, startable rows): model MAE **4.87** vs trailing-4 5.37
  (**+9.2%**), beats the naive baselines at all four positions. External
  sources (Sleeper, ESPN, Vegas props) are blended through a leak-free
  in-season **bias calibrator** (`fantasy/projections/calibration.py`) — the
  calibrated consensus reached **4.81 MAE** on 2025, better than any single
  source. Upcoming (unplayed) weeks are projected via synthesized
  point-in-time rows (`features.future_frame`), with bye teams excluded.
- **Phase 2** — notify loop (FastAPI + scheduler), news + expert signals.
- **Phase 3** — write execution behind the approval gate (deep-link tier live;
  browser tier needs the 2026 season for selector verification).
- **Phase 4** — draft engine (offline, self-play validated); live poller TBD.
- **Phase 5** — dashboard: lineup (injury/bye-aware), waivers with FAAB bids,
  trade ideas + analyzer, standings, chat, season report card.

Retrain/backtest anytime: `uv run python scripts/backtest_projections.py`.

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

## Trade Analyzer

The dashboard's Trade Analyzer evaluates **any multi-player trade** (2-for-1,
3-for-2, …) against *your* live roster and league rules — not by adding up player
values, but by the change in your best legal **starting lineup** over the rest of
the season. Because a lineup only rewards players who fill a slot, two players who
can't both start are worth less than one "big hitter" who takes a slot; roster fit
falls out of the math instead of being hand-tuned.

Alongside the headline lineup gain it reports the raw-points swing (to show what
never reaches your lineup), cross-positional value (VOR) fairness, a depth /
bench-insurance term (thinning depth has a cost), roster-legality warnings
(IR-aware "you'd need to drop N"), and — for a single-team offer — the opponent's
estimated accept-likelihood. Players are chosen from searchable dropdowns ("give"
scoped to your roster, "get" to other teams) so a typo can't break a lookup.

Core logic: `fantasy/decisions/trades.py::evaluate_trade_package` (shares the greedy
lineup engine in `decisions/lineup.py` with the start/sit optimizer). It runs off a
`trade` block baked into the dashboard snapshot (`api/dashboard_data.py`), so
analysis is instant and needs no live ESPN/model call.

**API** — `POST /api/analyze-trade` (Clerk-authenticated):

```jsonc
// request
{ "league": "<league-uuid>", "give": ["<player_id>", ...], "get": ["<player_id>", ...] }

// response (or { "error": "<reason>" })
{
  "lineup_delta": 6.4,        // ROS starting-lineup gain — the headline number
  "points_sum_delta": 14.2,   // raw ROS points swing (get minus give)
  "vor_delta": 5.1, "fairness": "slightly lopsided",
  "depth_delta": -1.3, "adjusted_delta": 5.1,  // lineup_delta + depth (drives the verdict)
  "accept_prob": 0.63,        // single-counterparty offers only (else null)
  "need_to_drop": 0, "verdict": "favorable",
  "give": [/* {id,name,pos,ros_proj,ros_vor,starter} … */],
  "get":  [/* … */],
  "notes": ["…"]              // plain-English caveats (bench glut, drops needed, …)
}
```

The analyzer activates once a league is connected and its analysis is built;
before that the panel shows a "connect ESPN & build" prompt.

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

