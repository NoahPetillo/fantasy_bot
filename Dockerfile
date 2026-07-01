# syntax=docker/dockerfile:1

# ── Fantasy App — production image for Render / Railway / Fly.io ───────────────
# This runs the app as a normal long-running FastAPI process (NOT serverless):
# it serves the dashboard + chatbot and, when ESPN cookies are present, runs the
# advise-only scheduler. A persistent disk holds the SQLite store and snapshots.
FROM python:3.12-slim

# uv: fast, reproducible installs straight from uv.lock.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# libgomp1 is the OpenMP runtime that xgboost/lightgbm load at import time; the
# slim base image doesn't ship it, so install it or model training will crash.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    HOST=0.0.0.0 \
    DATA_DIR=/data

WORKDIR /app

# 1) Dependency layer — cached until pyproject/uv.lock change. README is required
#    because hatchling reads it from the project metadata.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# 2) App layer — copy source, then install the `fantasy` package itself.
COPY fantasy ./fantasy
COPY config ./config
COPY scripts ./scripts
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Persistent state (SQLite store, snapshots, generated media) lives at $DATA_DIR,
# backed by a mounted disk in production. Create it so the app boots even before a
# volume is attached.
RUN mkdir -p /data
VOLUME ["/data"]

# Render/Railway inject $PORT; serve.py binds $HOST:$PORT and starts the scheduler
# when ESPN cookies are configured.
CMD [".venv/bin/python", "scripts/serve.py"]
