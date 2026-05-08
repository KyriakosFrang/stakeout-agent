from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone

from stakeout_agent.backends.base import AbstractMonitorDB

_log = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE = 0.5  # seconds; doubles each attempt

_CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id                      TEXT PRIMARY KEY,
    graph_id                    TEXT,
    thread_id                   TEXT,
    status                      TEXT DEFAULT 'running',
    started_at                  TIMESTAMPTZ DEFAULT NOW(),
    ended_at                    TIMESTAMPTZ,
    error                       TEXT,
    total_input_tokens          INTEGER,
    total_output_tokens         INTEGER,
    estimated_cost_usd          DOUBLE PRECISION,
    total_cache_read_tokens     INTEGER,
    total_cache_creation_tokens INTEGER
);

CREATE TABLE IF NOT EXISTS events (
    id                   SERIAL PRIMARY KEY,
    run_id               TEXT,
    graph_id             TEXT,
    event_type           TEXT,
    node_name            TEXT,
    latency_ms           DOUBLE PRECISION,
    payload              JSONB,
    error                TEXT,
    messages             JSONB,
    input_tokens         INTEGER,
    output_tokens        INTEGER,
    model                TEXT,
    timestamp            TIMESTAMPTZ DEFAULT NOW(),
    cache_read_tokens    INTEGER,
    cache_creation_tokens INTEGER
);

CREATE INDEX IF NOT EXISTS idx_runs_started_at  ON runs(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_runs_graph_id    ON runs(graph_id);
CREATE INDEX IF NOT EXISTS idx_runs_status      ON runs(status);
CREATE INDEX IF NOT EXISTS idx_events_run_id    ON events(run_id);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp DESC);

ALTER TABLE runs   ADD COLUMN IF NOT EXISTS total_input_tokens          INTEGER;
ALTER TABLE runs   ADD COLUMN IF NOT EXISTS total_output_tokens         INTEGER;
ALTER TABLE runs   ADD COLUMN IF NOT EXISTS estimated_cost_usd          DOUBLE PRECISION;
ALTER TABLE runs   ADD COLUMN IF NOT EXISTS total_cache_read_tokens     INTEGER;
ALTER TABLE runs   ADD COLUMN IF NOT EXISTS total_cache_creation_tokens INTEGER;
ALTER TABLE events ADD COLUMN IF NOT EXISTS input_tokens                INTEGER;
ALTER TABLE events ADD COLUMN IF NOT EXISTS output_tokens               INTEGER;
ALTER TABLE events ADD COLUMN IF NOT EXISTS model                       TEXT;
ALTER TABLE events ADD COLUMN IF NOT EXISTS llm_input                   JSONB;
ALTER TABLE events ADD COLUMN IF NOT EXISTS llm_output                  TEXT;
ALTER TABLE events ADD COLUMN IF NOT EXISTS cache_read_tokens           INTEGER;
ALTER TABLE events ADD COLUMN IF NOT EXISTS cache_creation_tokens       INTEGER;
"""

_schema_initialized = False
_schema_init_lock = threading.Lock()

# Retryable psycopg2 error class names — checked by name so this module
# stays importable even when psycopg2 is not installed.
_RETRYABLE_PG_EXC_NAMES = frozenset({"OperationalError", "InterfaceError"})


def _is_retryable(exc: Exception) -> bool:
    module = type(exc).__module__ or ""
    return module.startswith("psycopg2") and type(exc).__name__ in _RETRYABLE_PG_EXC_NAMES


def _make_pg_conn():
    try:
        import psycopg2
    except ImportError as exc:
        raise ImportError(
            "psycopg2 is required for the PostgreSQL backend. Install it with: pip install 'stakeout-agent[postgres]'"
        ) from exc

    uri = os.getenv("POSTGRES_URI") or os.getenv("DATABASE_URL", "postgresql://localhost/stakeout")
    conn = psycopg2.connect(uri, connect_timeout=5)
    conn.autocommit = True
    global _schema_initialized
    if not _schema_initialized:
        with _schema_init_lock:
            if not _schema_initialized:
                with conn.cursor() as cur:
                    cur.execute(_CREATE_TABLES_SQL)
                _schema_initialized = True
    _log.debug("PostgresMonitorDB connected uri=%s", uri)
    return conn


class PostgresMonitorDB(AbstractMonitorDB):
    def __init__(self):
        self._conn = None
        self._lock = threading.Lock()

    @property
    def _connection(self):
        if self._conn is None or self._conn.closed:
            with self._lock:
                if self._conn is None or self._conn.closed:  # double-checked locking
                    self._conn = _make_pg_conn()
        return self._conn

    def _reset_conn(self) -> None:
        with self._lock:
            self._conn = None

    def _run_with_retry(self, op_name: str, fn) -> None:
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                fn()
                return
            except ImportError:
                raise  # psycopg2 not installed — programming error, not transient
            except Exception as exc:
                if _is_retryable(exc):
                    self._reset_conn()
                    if attempt < _MAX_RETRIES:
                        delay = _RETRY_BACKOFF_BASE * (2 ** (attempt - 1))
                        _log.warning(
                            "%s attempt %d/%d failed: %s — retrying in %.1fs",
                            op_name,
                            attempt,
                            _MAX_RETRIES,
                            exc,
                            delay,
                        )
                        time.sleep(delay)
                    else:
                        _log.error("%s failed after %d attempts: %s", op_name, _MAX_RETRIES, exc)
                else:
                    # Non-retryable error; reset the connection if it was closed by the failure.
                    if self._conn is not None and self._conn.closed:
                        self._reset_conn()
                    _log.error("%s failed: %s", op_name, exc)
                    return

    def create_run(self, run_id: str, graph_id: str, thread_id: str) -> None:
        def _op():
            with self._connection.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO runs (run_id, graph_id, thread_id, status, started_at, ended_at, error)
                    VALUES (%s, %s, %s, 'running', %s, NULL, NULL)
                    """,
                    (run_id, graph_id, thread_id, datetime.now(timezone.utc)),
                )
            _log.debug("create_run inserted run_id=%s graph_id=%s", run_id, graph_id)

        self._run_with_retry(f"create_run {run_id}", _op)

    def complete_run(
        self,
        run_id: str,
        total_input_tokens: int | None = None,
        total_output_tokens: int | None = None,
        estimated_cost_usd: float | None = None,
        total_cache_read_tokens: int | None = None,
        total_cache_creation_tokens: int | None = None,
    ) -> None:
        def _op():
            with self._connection.cursor() as cur:
                cur.execute(
                    """
                    UPDATE runs
                    SET status = 'completed', ended_at = %s,
                        total_input_tokens = %s, total_output_tokens = %s, estimated_cost_usd = %s,
                        total_cache_read_tokens = %s, total_cache_creation_tokens = %s
                    WHERE run_id = %s
                    """,
                    (
                        datetime.now(timezone.utc),
                        total_input_tokens,
                        total_output_tokens,
                        estimated_cost_usd,
                        total_cache_read_tokens,
                        total_cache_creation_tokens,
                        run_id,
                    ),
                )
                if cur.rowcount == 0:
                    _log.warning("complete_run: no run found with id %s", run_id)
                else:
                    _log.debug("complete_run run_id=%s", run_id)

        self._run_with_retry(f"complete_run {run_id}", _op)

    def fail_run(self, run_id: str, error: str) -> None:
        def _op():
            with self._connection.cursor() as cur:
                cur.execute(
                    "UPDATE runs SET status = 'failed', ended_at = %s, error = %s WHERE run_id = %s",
                    (datetime.now(timezone.utc), error, run_id),
                )
                if cur.rowcount == 0:
                    _log.warning("fail_run: no run found with id %s", run_id)
                else:
                    _log.debug("fail_run run_id=%s", run_id)

        self._run_with_retry(f"fail_run {run_id}", _op)

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
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        model: str | None = None,
        llm_input: list[dict] | None = None,
        llm_output: str | None = None,
        cache_read_tokens: int | None = None,
        cache_creation_tokens: int | None = None,
    ) -> None:
        def _op():
            with self._connection.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO events
                        (run_id, graph_id, event_type, node_name, latency_ms, payload, error,
                         messages, input_tokens, output_tokens, model, llm_input, llm_output,
                         cache_read_tokens, cache_creation_tokens, timestamp)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                        input_tokens,
                        output_tokens,
                        model,
                        json.dumps(llm_input) if llm_input is not None else None,
                        llm_output,
                        cache_read_tokens,
                        cache_creation_tokens,
                        datetime.now(timezone.utc),
                    ),
                )
            _log.debug("insert_event event_type=%s node=%s run_id=%s", event_type, node_name, run_id)

        self._run_with_retry(f"insert_event {run_id}", _op)
