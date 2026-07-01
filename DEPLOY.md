# Deploying the Fantasy App online

This app is a **long-running FastAPI service** (it serves the dashboard + chatbot
and runs a background scheduler when ESPN cookies are present). That rules out
serverless platforms like Vercel — the ML stack is too big for the size limit, the
scheduler needs an always-on process, and the SQLite store needs a real disk.

The easy, correct host is a **persistent container host**: Render or Railway. The
included `Dockerfile` works on both; `render.yaml` is a Render convenience.

---

## Access model (your password)

- Set **`SITE_PASSWORD`** → the whole dashboard locks behind a login screen.
- The **chatbot stays open** — league mates use it without the password.
- Leave `SITE_PASSWORD` unset → the site is fully open (fine for local dev).

Auth is a signed HttpOnly cookie (30 days). Changing `SITE_PASSWORD` instantly
logs everyone out. See `fantasy/api/auth.py`.

---

## Environment variables

| Var | Required? | What it does |
|-----|-----------|--------------|
| `SITE_PASSWORD` | **For privacy** | Locks every feature except the chatbot. |
| `CHAT_RATE_LIMIT` | Optional | Max chatbot questions per visitor IP per hour (default `250`). The logged-in owner is exempt. Set `0` to disable. |
| `GROQ_API_KEY` or `ANTHROPIC_API_KEY` | Recommended | Powers the chatbot's free-form answers. Without a key it uses a keyless fallback parser. |
| `ESPN_S2`, `ESPN_SWID`, `ESPN_LEAGUE_ID` | For live data | Live ESPN reads + the advise scheduler. Without them the app serves snapshots only. |
| `ESPN_SEASON`, `ESPN_TEAM_ID` | Optional | Defaults to 2025 / first team. |
| `DISCORD_WEBHOOK_URL` | Optional | Where approved "moments" post. |
| `HOST` | Set by Docker | `0.0.0.0` in the container (already set). |
| `DATA_DIR` | Set by Docker | `/data` — the mounted disk (already set). |
| `DATABASE_URL` | **Multi-tenant** | Neon Postgres URL. All per-user state lives here. Unset → local SQLite fallback (single-box dev only). |
| `CREDENTIAL_ENC_KEY` | **Multi-tenant** | Fernet key encrypting users' ESPN cookies at rest. Generate: `python -m fantasy.security.crypto`. Keep it in the host secret store, never in the DB. |
| `CLERK_PUBLISHABLE_KEY`, `CLERK_SECRET_KEY` | **Multi-tenant** | Clerk (managed auth). Publishable for the frontend, secret for backend Clerk API calls. |
| `CLERK_ISSUER` | Optional | Pin the Clerk issuer/JWKS; otherwise derived from the session token. |

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

## Option 1 — Render (recommended, uses `render.yaml`)

1. Push this repo to GitHub.
2. Render dashboard → **New + → Blueprint** → pick the repo. It reads
   `render.yaml`, builds the `Dockerfile`, and attaches a 1 GB disk at `/data`.
3. When prompted, fill in the secret env vars (at minimum `SITE_PASSWORD` and one
   LLM key; add the ESPN cookies for live data).
4. Deploy. First build takes a few minutes (the ML wheels are large). When it's
   live, Render gives you a `https://fantasy-app-xxxx.onrender.com` URL.

Plan: **starter** (~$7/mo, 512 MB) is enough to serve the dashboard + chatbot.
If you add ESPN cookies and let it train the projection model on startup, bump to
**standard** (2 GB) in `render.yaml` or the dashboard.

## Option 2 — Railway (uses the `Dockerfile` directly)

1. Push to GitHub. Railway → **New Project → Deploy from GitHub repo**. It detects
   the `Dockerfile` automatically.
2. **Variables** tab → add the env vars from the table (`HOST=0.0.0.0`,
   `DATA_DIR=/data`, `SITE_PASSWORD`, an LLM key, ESPN cookies).
3. **Volumes** → add a volume mounted at `/data` (so SQLite + snapshots persist).
4. Deploy; Railway gives you a public URL under **Settings → Networking**.

---

## After it's live

- Visit the URL: you should see the **lock screen**. Enter `SITE_PASSWORD` → the
  dashboard loads. The chat bubble works without logging in.
- Health check: `GET /health` returns `{"status":"ok", ...}`.
- No data yet? The dashboard shows a fallback until a snapshot is built. With ESPN
  cookies set, add a league in the sidebar and hit **Build full analysis**.

---

## ⚠️ Two things to know before sharing the link

1. **The chatbot is public and rate-limited** to `CHAT_RATE_LIMIT` questions per
   visitor IP per hour (default 250); you (logged in) are exempt. That caps abuse
   from any single visitor. On a free-tier LLM key there's no bill risk anyway —
   tune the number or set `0` to disable.
2. **Your secrets are sensitive.** ESPN cookies grant access to your ESPN account;
   LLM keys cost money. They live as host env vars (never in the image — `.env` is
   gitignored and dockerignored). Don't commit them; rotate if leaked.
