# Deploying the Fantasy App online

This app is a **long-running FastAPI service** (it serves the dashboard + chatbot
and runs a background scheduler when ESPN cookies are present). That rules out
serverless platforms like Vercel — the ML stack is too big for the size limit, the
scheduler needs an always-on process, and the SQLite store needs a real disk.

The easy, correct host is a **persistent container host**: Render or Railway. The
included `Dockerfile` works on both; `render.yaml` is a Render convenience.

---

## Access model (Clerk)

Auth is **Clerk** (managed) — the shared-password gate was removed. Users sign up /
sign in via Clerk; the backend verifies the session JWT and scopes every request to
that user. Plans are **Free** vs **Pro** (Stripe), gating chat quota + league count.

---

## Environment variables

| Var | Required? | What it does |
|-----|-----------|--------------|
| `CHAT_RATE_LIMIT` | Optional | Per-IP chatbot abuse floor per hour (default `250`). Per-user plan quotas are enforced separately. Set `0` to disable the floor. |
| `GROQ_API_KEY` or `ANTHROPIC_API_KEY` | Recommended | Powers the chatbot's free-form answers. Without a key it uses a keyless fallback parser. |
| `ESPN_S2`, `ESPN_SWID`, `ESPN_LEAGUE_ID` | For live data | Live ESPN reads + the advise scheduler. Without them the app serves snapshots only. |
| `ESPN_SEASON`, `ESPN_TEAM_ID` | Optional | Defaults to 2025 / first team. |
| `DISCORD_WEBHOOK_URL` | Optional | Where approved "moments" post. |
| `HOST` | Set by Docker | `0.0.0.0` in the container (already set). |
| `DATA_DIR` | Set by Docker | `/data` — the mounted disk (already set). |
| `DATABASE_URL` | **Multi-tenant** | Neon Postgres URL. All per-user state lives here. Unset → local SQLite fallback (single-box dev only). |
| `CREDENTIAL_ENC_KEY` | **Multi-tenant** | Fernet key encrypting users' ESPN cookies at rest. Generate: `python -m fantasy.security.crypto`. Keep it in the host secret store, never in the DB. |
| `CLERK_PUBLISHABLE_KEY`, `CLERK_SECRET_KEY` | **Multi-tenant** | Clerk (managed auth). Publishable for the frontend, secret for backend Clerk API calls. |
| `CLERK_ISSUER` | Optional | Pin the Clerk issuer/JWKS; otherwise derived from the publishable key. |
| `STRIPE_SECRET_KEY`, `STRIPE_WEBHOOK_SECRET` | For paid plans | Stripe subscriptions. Unset → billing disabled (everyone stays on Free). |
| `STRIPE_PRICE_ID` | For paid plans | The recurring price (`price_…`) of the Pro plan created in Stripe. |

Get your `ESPN_S2` / `ESPN_SWID` from the cookies on espn.com while logged in
(same values as your local `.env`).

---

## Multi-tenant infrastructure (Neon + Clerk + Fernet)

This app is migrating from single-tenant to a multi-tenant SaaS (see
`MULTITENANT_BUILD.md`). Per-user state moves from local files into **Neon
Postgres**, auth is handled by **Clerk**, and each user's ESPN cookies are
**Fernet-encrypted** with a key kept outside the database.

1. **Neon** → create a project (region: **US East**) and copy the connection
   string into `DATABASE_URL`. (SQLAlchemy auto-normalizes `postgres://` →
   `postgresql+psycopg://`.)
2. **Fernet key** → `python -m fantasy.security.crypto` prints a key; set it as
   `CREDENTIAL_ENC_KEY`. Losing this key makes stored cookies unrecoverable
   (users just reconnect). It supports comma-separated keys for rotation.
3. **Clerk** → create an application, then set `CLERK_PUBLISHABLE_KEY` and
   `CLERK_SECRET_KEY`.
4. **Migrations** run automatically on Render via `preDeployCommand`
   (`alembic upgrade head`). To run them manually: `alembic upgrade head`.

Once on Postgres the app no longer needs the persistent disk for *state* (the
disk is only for cached model data / generated media). See `MULTITENANT_BUILD.md`
for the phased plan and hard requirements.

---

## Billing (Stripe — optional; enables the Pro plan)

Without Stripe config everyone stays on **Free** (25 chat questions/day, 1 league).
To enable **Pro** (1,000/day, 10 leagues — tunable in `fantasy/billing/plans.py`):

1. In Stripe, create a **recurring Product/Price** for Pro and copy its price id
   (`price_…`) → `STRIPE_PRICE_ID`.
2. Set `STRIPE_SECRET_KEY` (use `sk_test_…` until you go live).
3. Add a **webhook endpoint** pointing at `https://<your-app>/api/stripe/webhook`,
   subscribe to `checkout.session.completed` and `customer.subscription.*`, and set
   the signing secret as `STRIPE_WEBHOOK_SECRET`.

Checkout + the customer billing portal are created server-side; the webhook keeps
`users.plan` in sync. Quotas are enforced per user (`chat_usage` table) with the
per-IP limiter as an anonymous floor.

---

## Option 1 — Render (recommended, uses `render.yaml`)

1. Push this repo to GitHub.
2. Render dashboard → **New + → Blueprint** → pick the repo. It reads
   `render.yaml`, builds the `Dockerfile`, and attaches a 1 GB disk at `/data`.
3. When prompted, fill in the secret env vars (at minimum `DATABASE_URL`,
   `CREDENTIAL_ENC_KEY`, the `CLERK_*` keys, and one LLM key; add `STRIPE_*` for paid
   plans).
4. Deploy. First build takes a few minutes (the ML wheels are large). When it's
   live, Render gives you a `https://fantasy-app-xxxx.onrender.com` URL.

Plan: **starter** (~$7/mo, 512 MB) is enough to serve the dashboard + chatbot.
If you add ESPN cookies and let it train the projection model on startup, bump to
**standard** (2 GB) in `render.yaml` or the dashboard.

## Option 2 — Railway (uses the `Dockerfile` directly)

1. Push to GitHub. Railway → **New Project → Deploy from GitHub repo**. It detects
   the `Dockerfile` automatically.
2. **Variables** tab → add the env vars from the table (`HOST=0.0.0.0`,
   `DATA_DIR=/data`, `DATABASE_URL`, `CREDENTIAL_ENC_KEY`, the `CLERK_*` keys, an
   LLM key; add `STRIPE_*` for paid plans).
3. **Migrations** → run `alembic upgrade head` once against `DATABASE_URL` (Render
   does this automatically via `preDeployCommand`; on Railway add it as a deploy/
   release command or run it once from a shell).
4. **Volumes** → add a volume mounted at `/data` (only for cached model data /
   generated media now — per-user state lives in Postgres).
5. Deploy; Railway gives you a public URL under **Settings → Networking**.

---

## After it's live

- Visit the URL: you should see the **Clerk sign-in**. Sign up / sign in → the
  dashboard loads, scoped to your account. The public chat bubble works without an
  account (rate-limited per IP).
- Health check: `GET /health` returns `{"status":"ok", ...}`; `GET /api/config`
  returns `auth_configured: true` once the `CLERK_*` keys are set.
- New account: open **Settings → Connect ESPN**, paste your `espn_s2` / `SWID`
  cookies (consent required), add a league, then hit **Build**. Preseason leagues
  show a shell view until the season's stats publish.

---

## ⚠️ Two things to know before sharing the link

1. **The chatbot is public and rate-limited** to `CHAT_RATE_LIMIT` questions per
   visitor IP per hour (default 250); you (logged in) are exempt. That caps abuse
   from any single visitor. On a free-tier LLM key there's no bill risk anyway —
   tune the number or set `0` to disable.
2. **Your secrets are sensitive.** ESPN cookies grant access to your ESPN account;
   LLM keys cost money. They live as host env vars (never in the image — `.env` is
   gitignored and dockerignored). Don't commit them; rotate if leaked.
