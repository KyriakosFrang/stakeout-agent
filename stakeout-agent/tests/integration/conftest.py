from __future__ import annotations

import uuid

import pytest


@pytest.fixture(scope="session")
def _mongo_reachable():
    try:
        from pymongo import MongoClient

        client = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=2_000)
        client.admin.command("ping")
        client.close()
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def _postgres_reachable():
    try:
        import psycopg2

        conn = psycopg2.connect(
            "postgresql://stakeout:stakeout@localhost/stakeout",
            connect_timeout=2,
        )
        conn.close()
        return True
    except Exception:
        return False


@pytest.fixture
def mongo_db(_mongo_reachable):
    if not _mongo_reachable:
        pytest.skip("MongoDB not available — run `docker compose up -d mongo`")
    from stakeout_agent.backends.mongodb import MongoMonitorDB

    return MongoMonitorDB()


_DEFAULT_POSTGRES_URI = "postgresql://stakeout:stakeout@localhost/stakeout"


@pytest.fixture
def pg_db(_postgres_reachable):
    if not _postgres_reachable:
        pytest.skip("PostgreSQL not available — run `docker compose up -d postgres`")
    import os

    os.environ.setdefault("POSTGRES_URI", _DEFAULT_POSTGRES_URI)
    from stakeout_agent.backends.postgres import PostgresMonitorDB

    return PostgresMonitorDB()


@pytest.fixture
def run_id():
    return str(uuid.uuid4())
