from __future__ import annotations

import json
import logging
import os
import statistics
import threading
import time
from datetime import datetime, timedelta, timezone

from stakeout_agent.backends.base import AbstractMonitorDB, AbstractQueryDB
from stakeout_agent.retention import RetentionPolicy

_log = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE = 0.5  # seconds; doubles each attempt

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

    from stakeout_agent.backends.migrations import run_migrations

    run_migrations(uri)

    conn = psycopg2.connect(uri, connect_timeout=5)
    conn.autocommit = True
    _log.debug("PostgresMonitorDB connected uri=%s", uri)
    return conn


_RUN_COLS = (
    "run_id", "graph_id", "thread_id", "status", "started_at", "ended_at", "error",
    "run_inputs", "parent_run_id", "prompt_version",
    "total_input_tokens", "total_output_tokens", "estimated_cost_usd",
    "total_cache_read_tokens", "total_cache_creation_tokens",
)

_EVENT_COLS = (
    "run_id", "graph_id", "event_type", "node_name", "latency_ms",
    "timestamp", "error", "input_tokens", "output_tokens", "model",
    "llm_output", "cache_read_tokens", "cache_creation_tokens",
)

_RUN_SELECT = f"SELECT {', '.join(_RUN_COLS)} FROM runs"
_EVENT_SELECT = f"SELECT {', '.join(_EVENT_COLS)} FROM events"


def _row_to_run(cols: tuple, row: tuple) -> dict:
    d = dict(zip(cols, row))
    for field in ("started_at", "ended_at"):
        if d.get(field) is not None:
            d[field] = d[field].isoformat()
    started = row[cols.index("started_at")] if "started_at" in cols else None
    ended = row[cols.index("ended_at")] if "ended_at" in cols else None
    if started and ended:
        d["latency_ms"] = (ended - started).total_seconds() * 1000
    else:
        d["latency_ms"] = None
    return d


def _row_to_event(cols: tuple, row: tuple) -> dict:
    d = dict(zip(cols, row))
    if d.get("timestamp") is not None:
        d["timestamp"] = d["timestamp"].isoformat()
    return d


def _percentile(values: list[float], p: int) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100)[p - 1]


class PostgresMonitorDB(AbstractMonitorDB, AbstractQueryDB):
    def __init__(self, retention: RetentionPolicy | None = None):
        self._conn = None
        self._lock = threading.Lock()
        self._retention = retention
        self._run_expires: dict[str, datetime] = {}

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

    def create_run(
        self,
        run_id: str,
        graph_id: str,
        thread_id: str,
        run_inputs: str | None = None,
        parent_run_id: str | None = None,
        prompt_version: str | None = None,
        environment: str | None = None,
    ) -> None:
        exp_at = self._retention.expires_at(graph_id=graph_id, environment=environment) if self._retention else None
        if exp_at is not None:
            self._run_expires[run_id] = exp_at

        def _op():
            with self._connection.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO runs
                        (run_id, graph_id, thread_id, status, started_at, ended_at, error,
                         run_inputs, parent_run_id, prompt_version, expires_at)
                    VALUES (%s, %s, %s, 'running', %s, NULL, NULL, %s, %s, %s, %s)
                    """,
                    (
                        run_id,
                        graph_id,
                        thread_id,
                        datetime.now(timezone.utc),
                        run_inputs,
                        parent_run_id,
                        prompt_version,
                        exp_at,
                    ),
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
        self._run_expires.pop(run_id, None)

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
        self._run_expires.pop(run_id, None)

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

    def prune_runs(self, older_than_days: int) -> int:
        cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
        deleted = 0

        def _op():
            nonlocal deleted
            with self._connection.cursor() as cur:
                cur.execute(
                    "DELETE FROM events WHERE run_id IN (SELECT run_id FROM runs WHERE started_at < %s)",
                    (cutoff,),
                )
                cur.execute("DELETE FROM runs WHERE started_at < %s", (cutoff,))
                deleted = cur.rowcount
                _log.info("prune_runs deleted %d runs older than %d days", deleted, older_than_days)

        self._run_with_retry(f"prune_runs older_than_days={older_than_days}", _op)
        return deleted

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
        exp_at = self._run_expires.get(run_id)

        def _op():
            with self._connection.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO events
                        (run_id, graph_id, event_type, node_name, latency_ms, payload, error,
                         messages, input_tokens, output_tokens, model, llm_input, llm_output,
                         cache_read_tokens, cache_creation_tokens, timestamp, expires_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                        exp_at,
                    ),
                )
            _log.debug("insert_event event_type=%s node=%s run_id=%s", event_type, node_name, run_id)

        self._run_with_retry(f"insert_event {run_id}", _op)

    # ------------------------------------------------------------------
    # AbstractQueryDB
    # ------------------------------------------------------------------

    def query_recent_runs(self, graph_id: str | None, limit: int) -> list[dict]:
        with self._connection.cursor() as cur:
            if graph_id:
                cur.execute(f"{_RUN_SELECT} WHERE graph_id = %s ORDER BY started_at DESC LIMIT %s", (graph_id, limit))
            else:
                cur.execute(f"{_RUN_SELECT} ORDER BY started_at DESC LIMIT %s", (limit,))
            cols = tuple(d.name for d in cur.description)
            return [_row_to_run(cols, row) for row in cur.fetchall()]

    def query_run_detail(self, run_id: str) -> dict | None:
        with self._connection.cursor() as cur:
            cur.execute(f"{_RUN_SELECT} WHERE run_id = %s", (run_id,))
            run_row = cur.fetchone()
            if run_row is None:
                return None
            run_cols = tuple(d.name for d in cur.description)
            cur.execute(f"{_EVENT_SELECT} WHERE run_id = %s ORDER BY timestamp ASC", (run_id,))
            evt_cols = tuple(d.name for d in cur.description)
            events = [_row_to_event(evt_cols, row) for row in cur.fetchall()]
        return {"run": _row_to_run(run_cols, run_row), "events": events}

    def query_failed_runs(self, graph_id: str | None, since_ts: float) -> list[dict]:
        since_dt = datetime.fromtimestamp(since_ts, tz=timezone.utc)
        with self._connection.cursor() as cur:
            if graph_id:
                cur.execute(
                    f"{_RUN_SELECT} WHERE status = 'failed' AND started_at >= %s"
                    f" AND graph_id = %s ORDER BY started_at DESC",
                    (since_dt, graph_id),
                )
            else:
                cur.execute(
                    f"{_RUN_SELECT} WHERE status = 'failed' AND started_at >= %s ORDER BY started_at DESC",
                    (since_dt,),
                )
            cols = tuple(d.name for d in cur.description)
            return [_row_to_run(cols, row) for row in cur.fetchall()]

    def query_slow_runs(self, graph_id: str | None, threshold_ms: float, since_ts: float) -> list[dict]:
        since_dt = datetime.fromtimestamp(since_ts, tz=timezone.utc)
        threshold_s = threshold_ms / 1000.0
        with self._connection.cursor() as cur:
            if graph_id:
                cur.execute(
                    f"{_RUN_SELECT} WHERE status = 'completed' AND started_at >= %s AND ended_at IS NOT NULL"
                    f" AND EXTRACT(EPOCH FROM (ended_at - started_at)) > %s AND graph_id = %s ORDER BY started_at DESC",
                    (since_dt, threshold_s, graph_id),
                )
            else:
                cur.execute(
                    f"{_RUN_SELECT} WHERE status = 'completed' AND started_at >= %s AND ended_at IS NOT NULL"
                    f" AND EXTRACT(EPOCH FROM (ended_at - started_at)) > %s ORDER BY started_at DESC",
                    (since_dt, threshold_s),
                )
            cols = tuple(d.name for d in cur.description)
            return [_row_to_run(cols, row) for row in cur.fetchall()]

    def query_run_stats(self, graph_id: str | None, since_ts: float) -> dict:
        since_dt = datetime.fromtimestamp(since_ts, tz=timezone.utc)
        with self._connection.cursor() as cur:
            if graph_id:
                cur.execute(f"{_RUN_SELECT} WHERE started_at >= %s AND graph_id = %s", (since_dt, graph_id))
            else:
                cur.execute(f"{_RUN_SELECT} WHERE started_at >= %s", (since_dt,))
            cols = tuple(d.name for d in cur.description)
            rows = [_row_to_run(cols, row) for row in cur.fetchall()]

        total = len(rows)
        failed = sum(1 for r in rows if r.get("status") == "failed")
        latencies = [ms for r in rows if (ms := r.get("latency_ms")) is not None]
        costs = [c for r in rows if (c := r.get("estimated_cost_usd")) is not None]
        return {
            "graph_id": graph_id,
            "run_count": total,
            "error_rate": (failed / total) if total else None,
            "p50_latency_ms": _percentile(latencies, 50),
            "p95_latency_ms": _percentile(latencies, 95),
            "total_cost_usd": sum(costs) if costs else None,
            "avg_cost_usd": (sum(costs) / len(costs)) if costs else None,
        }

    def query_runs_by_output(self, graph_id: str | None, text: str, limit: int) -> list[dict]:
        with self._connection.cursor() as cur:
            if graph_id:
                cur.execute(
                    f"{_RUN_SELECT} WHERE run_inputs ILIKE %s AND graph_id = %s ORDER BY started_at DESC LIMIT %s",
                    (f"%{text}%", graph_id, limit),
                )
            else:
                cur.execute(
                    f"{_RUN_SELECT} WHERE run_inputs ILIKE %s ORDER BY started_at DESC LIMIT %s",
                    (f"%{text}%", limit),
                )
            cols = tuple(d.name for d in cur.description)
            return [_row_to_run(cols, row) for row in cur.fetchall()]
