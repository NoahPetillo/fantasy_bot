"""Alembic environment.

Wired to the app: the URL comes from ``Settings.effective_database_url`` (Neon in
prod, local SQLite fallback in dev/CI) and the target metadata is the ORM's
``Base.metadata``, so ``alembic revision --autogenerate`` and ``alembic upgrade``
track the models in ``fantasy/db/models.py``.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from fantasy.config import settings
from fantasy.db.base import Base
import fantasy.db.models  # noqa: F401  (import registers all tables on Base.metadata)

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Inject the app's database URL (overrides any sqlalchemy.url in alembic.ini).
config.set_main_option("sqlalchemy.url", settings.effective_database_url)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=settings.effective_database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        render_as_batch=True,  # safe ALTERs on SQLite dev DBs
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            render_as_batch=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
