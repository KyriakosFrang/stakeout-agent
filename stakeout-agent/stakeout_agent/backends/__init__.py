from __future__ import annotations

import os

from stakeout_agent.backends.base import AbstractMonitorDB


def get_backend() -> AbstractMonitorDB:
    """Return the configured backend instance based on STAKEOUT_BACKEND env var.

    STAKEOUT_BACKEND=mongodb  (default) → MonitorDB (requires MONGO_URI / MONGO_DB)
    STAKEOUT_BACKEND=postgres           → PostgresMonitorDB (requires POSTGRES_URI or DATABASE_URL)
    """
    backend = os.getenv("STAKEOUT_BACKEND", "mongodb").lower()
    if backend == "postgres":
        from stakeout_agent.backends.postgres import PostgresMonitorDB

        return PostgresMonitorDB()
    from stakeout_agent.db import MonitorDB

    return MonitorDB()
