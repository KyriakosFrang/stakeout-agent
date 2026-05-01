from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone

from stakeout_agent.backends.base import AbstractMonitorDB

_log = logging.getLogger(__name__)

_CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    graph_id    TEXT,
    thread_id   TEXT,
    status      TEXT DEFAULT 'running',
    started_at  TIMESTAMPTZ DEFAULT NOW(),
    ended_at    TIMESTAMPTZ,
    error       TEXT
);

CREATE TABLE IF NOT EXISTS events (
    id          SERIAL PRIMARY KEY,
    run_id      TEXT,
    graph_id    TEXT,
    event_type  TEXT,
    node_name   TEXT,
    latency_ms  DOUBLE PRECISION,
    payload     JSONB,
    error       TEXT,
    messages    JSONB,
    timestamp   TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_runs_started_at  ON runs(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_runs_graph_id    ON runs(graph_id);
CREATE INDEX IF NOT EXISTS idx_runs_status      ON runs(status);
CREATE INDEX IF NOT EXISTS idx_events_run_id    ON events(run_id);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp DESC);
"""


def _make_pg_conn():
    try:
        import psycopg2
    except ImportError as exc:
        raise ImportError(
            "psycopg2 is required for the PostgreSQL backend. Install it with: pip install 'stakeout-agent[postgres]'"
        ) from exc

    uri = os.getenv("POSTGRES_URI") or os.getenv("DATABASE_URL", "postgresql://localhost/stakeout")
    conn = psycopg2.connect(uri)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(_CREATE_TABLES_SQL)
    _log.debug("PostgresMonitorDB connected uri=%s", uri)
    return conn


class PostgresMonitorDB(AbstractMonitorDB):
    def __init__(self):
        self._conn = None
        self._lock = threading.Lock()

    @property
    def _connection(self):
        if self._conn is None:
            with self._lock:
                if self._conn is None:  # double-checked locking
                    self._conn = _make_pg_conn()
        return self._conn

    def create_run(self, run_id: str, graph_id: str, thread_id: str) -> None:
        conn = self._connection  # propagates on connection failure, same as MonitorDB
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO runs (run_id, graph_id, thread_id, status, started_at, ended_at, error)
                    VALUES (%s, %s, %s, 'running', %s, NULL, NULL)
                    """,
                    (run_id, graph_id, thread_id, datetime.now(timezone.utc)),
                )
        except Exception as exc:
            _log.error("create_run %s failed: %s", run_id, exc)
            return
        _log.debug("create_run inserted run_id=%s graph_id=%s", run_id, graph_id)

    def complete_run(self, run_id: str) -> None:
        conn = self._connection
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE runs SET status = 'completed', ended_at = %s WHERE run_id = %s",
                    (datetime.now(timezone.utc), run_id),
                )
                if cur.rowcount == 0:
                    _log.warning("complete_run: no run found with id %s", run_id)
                else:
                    _log.debug("complete_run run_id=%s", run_id)
        except Exception as exc:
            _log.error("complete_run %s failed: %s", run_id, exc)

    def fail_run(self, run_id: str, error: str) -> None:
        conn = self._connection
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE runs SET status = 'failed', ended_at = %s, error = %s WHERE run_id = %s",
                    (datetime.now(timezone.utc), error, run_id),
                )
                if cur.rowcount == 0:
                    _log.warning("fail_run: no run found with id %s", run_id)
                else:
                    _log.debug("fail_run run_id=%s", run_id)
        except Exception as exc:
            _log.error("fail_run %s failed: %s", run_id, exc)

    def insert_event(
        self,
        run_id: str,
        graph_id: str,
        event_type: str,
        node_name: str,
        latency_ms: float | None = None,
        payload: dict | None = None,
        error: str | None = None,
        messages: list[dict] | None = None,
    ) -> None:
        conn = self._connection
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO events
                        (run_id, graph_id, event_type, node_name, latency_ms, payload, error, messages, timestamp)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        run_id,
                        graph_id,
                        event_type,
                        node_name,
                        latency_ms,
                        json.dumps(payload or {}),
                        error,
                        json.dumps(messages) if messages is not None else None,
                        datetime.now(timezone.utc),
                    ),
                )
        except Exception as exc:
            _log.error("insert_event for run %s failed: %s", run_id, exc)
            return
        _log.debug("insert_event event_type=%s node=%s run_id=%s", event_type, node_name, run_id)
