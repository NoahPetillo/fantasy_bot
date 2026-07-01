# Multi-tenant SaaS build plan — `fantasy_app`

> Handoff brief for a fresh Claude Code session working in this repo. Read this
> whole file first, then the files it points to, confirm the open decisions with
> the user, and build in phases. Keep the app bootable and the test suite green
> after every phase.

## Goal

Convert this **single-tenant** app (one user's ESPN cookies + league in `.env`,
state in local files) into a **multi-tenant SaaS**: anyone signs up, connects
their **own** ESPN account, adds their **own** leagues, and gets a private
dashboard, stats, and chatbot. End goal is a paid public product. Preserve the
existing "football brain" (projection/valuation/decision/draft/backtest modules) —
this refactor changes **tenancy, auth, and storage**, not the modeling.

## Locked stack (the user chose this — do not substitute without asking)

- **Database:** **Neon** (serverless Postgres, free tier to start). Access via
  **SQLAlchemy 2.0** + **Alembic** migrations.
- **Auth:** **Clerk** (managed). Do NOT hand-roll auth — password reset/email
  verification is where solo founders get breached. Frontend uses Clerk's
  hosted/embedded sign-up + sign-in; backend verifies the Clerk session token
  (JWKS) on each request and maps the Clerk user id → a `users` row. (Supabase
  Auth is an acceptable alternative only if the user asks.)
- **ESPN cookie encryption:** **Fernet** (`cryptography`). Key from env
  `CREDENTIAL_ENC_KEY`, held in the host's secret store — **never in the DB**.
- **Backend:** keep **FastAPI** (needed for server-side ESPN calls) + the existing
  static frontend; add sign-in, connect-ESPN, and settings screens.
- **Payments (Phase 5 only):** **Stripe**.

## ⚠️ Legal drafts already in the repo — READ THESE FIRST

The user and a prior session worked through the risk of storing other people's
ESPN cookies and wrote three drafts. **Read all three before touching the ESPN
credential flow** — they define the required UX and disclosures:

- **`legal/ESPN_CONNECT_CONSENT.md`** — the in-app consent screen. **The most
  important one.** This copy must appear at the "Connect ESPN" step behind a
  **required, unchecked-by-default checkbox**; block connecting until it's checked,
  and persist the consent (user id + version + timestamp).
- **`legal/PRIVACY.md`** — serve at `/privacy`.
- **`legal/TERMS.md`** — serve at `/terms`.

The reasoning behind them (so you don't weaken the guarantees): ESPN `espn_s2`/
`SWID` cookies are **session credentials** — whoever holds them can read a user's
private leagues and *could* make roster moves as them. So the non-negotiables are:
**encrypt at rest with the key separate from the DB**, **read-only to ESPN
always**, **never log/return cookies**, **explicit consent**, and **one-click
deletion**. Also surface a "not affiliated with or endorsed by ESPN/Disney"
disclaimer in the footer and the connect screen.

## Current architecture (read to understand what exists)

- `fantasy/config.py` — global settings incl. `espn_s2/swid/league_id` from `.env`.
- `fantasy/api/app.py` — FastAPI routes.
- `fantasy/api/auth.py` — current **single shared-password gate** (`SITE_PASSWORD`,
  signed HttpOnly cookie). This is what real per-user auth replaces.
- `fantasy/api/ratelimit.py` — per-IP chatbot rate limiter (keep as a floor).
- `fantasy/leagues.py` — `registry()` → global `data/leagues.json`.
- `fantasy/orchestrator/store.py` — `Store()` → global `data/fantasy.sqlite`.
- `fantasy/api/build.py` + `fantasy/api/dashboard_data.py` — build per-league
  snapshots to `data/dashboard_<league_id>.json`.
- `fantasy/espn/client.py` — `EspnClient`; reads cookies from global settings.
  Note it already accepts explicit `league_id/season/espn_s2/swid` args — use that.
- `fantasy/chat/` — chatbot over a league snapshot.
- `fantasy/api/static/dashboard.html` — single-page frontend.
- Modeling (keep as-is, only change how they get data): `fantasy/projections`,
  `fantasy/valuation`, `fantasy/decisions`, `fantasy/draft`, `fantasy/backtest`.

## Hard requirements (do not skip — verify each with a test)

1. **Per-user isolation.** Every query is scoped to the authenticated `user_id`,
   enforced at the query layer AND with DB foreign keys/constraints. Write an
   automated test proving user A cannot read or mutate user B's leagues,
   snapshots, or proposals through ANY endpoint.
2. **ESPN cookies are credentials.** Encrypt at rest (Fernet); key from
   `CREDENTIAL_ENC_KEY`, never in the DB. Decrypt only in-memory to build an
   `EspnClient` for that user's request. Never log, display, or return them;
   redact in error traces. Provide a "delete my ESPN credentials" endpoint and a
   "delete my account" endpoint (both immediate). Auto-purge credentials after
   repeated ESPN auth failures.
3. **Read-only to ESPN for everyone.** Never wire any write/execute path to a
   user's ESPN account. If `fantasy/execute/` is reachable, hard-gate it off for
   multi-tenant.
4. **Consent before storage.** Persist consent (version + timestamp) before
   accepting cookies; block the connect flow without it.
5. **Cost controls.** The chatbot now costs money per user — enforce a per-user
   (plan-based) quota; keep the IP rate limiter as a floor.
6. **App stays bootable and tests stay green after each phase.**

## Data model (Postgres)

- `users(id uuid pk, clerk_user_id unique, email, plan default 'free', created_at)`
- `espn_credentials(user_id fk unique, s2_enc, swid_enc, status, updated_at,
  consent_version, consent_at)`
- `leagues(id uuid pk, user_id fk, espn_league_id, team_id, season, name,
  created_at, unique(user_id, espn_league_id, season))`
- `snapshots(id uuid pk, league_id fk, week, payload jsonb, built_at)`  — replaces the JSON files
- `proposals(id uuid pk, user_id fk, league_id fk, kind, status, value, payload jsonb,
  created_at, decided_at)`  — replaces the SQLite `Store`
- `chat_usage(user_id fk, day date, count)`  — for per-user quota
- (Phase 5) `subscriptions(user_id fk, stripe_customer_id, plan, status, current_period_end)`

## Refactor map (single-tenant → per-user)

- Replace the `SITE_PASSWORD` gate with a Clerk-verified `current_user` FastAPI
  dependency used by every protected route.
- `registry()`/`leagues.json` → `leagues` table scoped to `user_id`.
- `Store()`/`fantasy.sqlite` → `proposals` table scoped to `user_id`.
- `dashboard_<id>.json` files → `snapshots` table (owned via league→user).
- `EspnClient` → construct per-request from the **current user's decrypted
  cookies** (pass `espn_s2`/`swid` explicitly), not from global settings.
- Chatbot → scope to the current user's league; enforce per-user quota. The old
  "public chatbot for league mates" concept goes away; chat is a logged-in feature.
- `fantasy/config.py` → remove reliance on per-user ESPN vars; keep only true
  app-level config (DB URL, enc key, Clerk keys, LLM keys, limits).

## Prerequisites the user provides (external accounts — you cannot create these)

You write all the code; the **user** creates the external SaaS accounts and hands
you the credentials. Prompt for these at the start of Phase 1, and provide safe
local `.env` placeholders so the app boots during development:

- **Neon:** the user creates a Postgres project at neon.tech and gives you the
  `DATABASE_URL`. Once you have it, YOU can run the Alembic migrations to create
  the tables in their database.
- **Clerk:** the user creates an application at clerk.com and gives you
  `CLERK_SECRET_KEY` + `CLERK_PUBLISHABLE_KEY` (and frontend/JWKS config).
- **Encryption key:** generate a Fernet key
  (`python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`);
  the user stores it as `CREDENTIAL_ENC_KEY` in their secret store. **Losing this
  key makes every stored ESPN cookie permanently undecryptable** — tell them to
  back it up.

## Phases (ship + test each before the next)

1. **Foundation.** Add Neon Postgres + SQLAlchemy + Alembic; models above;
   Fernet encryption util (+ round-trip test); Clerk auth wired to a `current_user`
   dependency; `users` upsert on first login. App still boots.
2. **Connect ESPN.** Build the connect screen from
   `legal/ESPN_CONNECT_CONSENT.md` (required checkbox + persisted consent);
   encrypt + store cookies; a "Test connection" button that validates against ESPN;
   wire `EspnClient` to read the current user's decrypted cookies; delete-credentials
   endpoint.
3. **Per-user data.** Move leagues + snapshots + proposals into Postgres, all
   scoped to `user_id`; port `build_shell`/`build_full`/dashboard/chat to per-user
   data. Add the cross-user isolation test.
4. **Frontend + legal.** Clerk sign-up/sign-in pages; connect-ESPN + settings
   (with delete controls); per-user dashboard; serve `/privacy` and `/terms` from
   the legal files; "not affiliated with ESPN" footer. Remove the shared-password gate.
5. **Monetize.** Stripe subscriptions + plan gating (free vs paid quotas/features);
   per-user chat quota enforced by plan.

## Deployment changes

- New env vars: `DATABASE_URL` (Neon), `CREDENTIAL_ENC_KEY` (Fernet key),
  `CLERK_SECRET_KEY`, `CLERK_PUBLISHABLE_KEY`; later `STRIPE_SECRET_KEY`,
  `STRIPE_WEBHOOK_SECRET`.
- State now lives in Postgres, so the app **no longer needs the persistent disk**
  for state (keep a disk only if you still generate media files). This also frees
  the deploy from the paid-disk requirement.
- Update `Dockerfile`, `render.yaml`, and `DEPLOY.md` accordingly. Run Alembic
  migrations on deploy (release step or startup).

## Confirm with the user before writing code

1. Neon for Postgres, Clerk for auth — confirmed? (Alternative: Supabase for both.)
2. Region for the DB / app.
3. Seed the existing single-user `.env` data as the first account, or start clean?
4. Keep the autonomous scheduler out of scope for now (on-demand builds only), to
   avoid per-user background-job complexity until after launch?

## Definition of done (v1)

A stranger can: sign up (Clerk) → connect their ESPN account behind the consent
screen → add a league → see their private dashboard + ask the chatbot → delete
their credentials/account — with full isolation from other users, cookies
encrypted at rest, `/privacy` and `/terms` live, and the isolation + encryption
tests passing.
