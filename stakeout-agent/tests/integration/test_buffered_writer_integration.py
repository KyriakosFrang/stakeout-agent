"""Integration tests for BufferedWriter wrapping real backends.

Requires running services:
  MongoDB  — docker compose up -d mongo
  Postgres — docker compose up -d postgres

Tests are automatically skipped when the target service is unreachable.
"""

from __future__ import annotations

import json
import time
import uuid
from unittest.mock import patch

import pytest

from stakeout_agent.writer import BufferedWriter

# ---------------------------------------------------------------------------
# Helpers shared across classes
# ---------------------------------------------------------------------------

_DEFAULT_POSTGRES_URI = "postgresql://stakeout:stakeout@localhost/stakeout"


def _unique_id() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# MongoDB — full lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestBufferedWriterMongo:
    """BufferedWriter wrapping MongoMonitorDB: writes must land in Mongo after close()."""

    def test_full_run_lifecycle_persists_to_mongo(self, mongo_db, tmp_path):
        run_id = _unique_id()
        try:
            with BufferedWriter(backend=mongo_db, dlq_path=str(tmp_path / "dlq.jsonl")) as writer:
                writer.create_run(run_id, "integration_graph", "thread-1", run_inputs='{"q": "hello"}')
                writer.insert_event(
                    run_id=run_id,
                    graph_id="integration_graph",
                    event_type="node_start",
                    node_name="agent",
                    payload={"inputs": "hello"},
                )
                writer.insert_event(
                    run_id=run_id,
                    graph_id="integration_graph",
                    event_type="node_end",
                    node_name="agent",
                    latency_ms=42.0,
                    input_tokens=10,
                    output_tokens=5,
                    model="claude-3",
                )
                writer.complete_run(run_id, total_input_tokens=10, total_output_tokens=5, estimated_cost_usd=0.0001)
            # close() has returned — all writes must be in Mongo

            run_doc = mongo_db.runs.find_one({"_id": run_id})
            assert run_doc is not None
            assert run_doc["status"] == "completed"
            assert run_doc["graph_id"] == "integration_graph"
            assert run_doc["thread_id"] == "thread-1"
            assert run_doc["total_input_tokens"] == 10
            assert run_doc["total_output_tokens"] == 5
            assert run_doc["ended_at"] is not None

            events = list(mongo_db.events.find({"run_id": run_id}).sort("timestamp", 1))
            assert len(events) == 2
            assert events[0]["event_type"] == "node_start"
            assert events[1]["event_type"] == "node_end"
            assert events[1]["latency_ms"] == pytest.approx(42.0)
            assert events[1]["model"] == "claude-3"
        finally:
            mongo_db.runs.delete_one({"_id": run_id})
            mongo_db.events.delete_many({"run_id": run_id})

    def test_failed_run_lifecycle_persists_to_mongo(self, mongo_db, tmp_path):
        run_id = _unique_id()
        try:
            with BufferedWriter(backend=mongo_db, dlq_path=str(tmp_path / "dlq.jsonl")) as writer:
                writer.create_run(run_id, "g", "t")
                writer.insert_event(run_id=run_id, graph_id="g", event_type="node_start", node_name="agent")
                writer.fail_run(run_id, "TimeoutError: upstream timed out")

            run_doc = mongo_db.runs.find_one({"_id": run_id})
            assert run_doc["status"] == "failed"
            assert run_doc["error"] == "TimeoutError: upstream timed out"
            assert run_doc["ended_at"] is not None
        finally:
            mongo_db.runs.delete_one({"_id": run_id})
            mongo_db.events.delete_many({"run_id": run_id})

    def test_concurrent_enqueues_all_land_in_mongo(self, mongo_db, tmp_path):
        """Many writes enqueued rapidly all arrive after close()."""
        run_id = _unique_id()
        n_events = 50
        try:
            with BufferedWriter(backend=mongo_db, dlq_path=str(tmp_path / "dlq.jsonl")) as writer:
                writer.create_run(run_id, "g", "t")
                for i in range(n_events):
                    writer.insert_event(
                        run_id=run_id,
                        graph_id="g",
                        event_type="node_start",
                        node_name=f"node_{i}",
                    )
                writer.complete_run(run_id)

            event_count = mongo_db.events.count_documents({"run_id": run_id})
            assert event_count == n_events
            assert writer.dropped_events == 0
        finally:
            mongo_db.runs.delete_one({"_id": run_id})
            mongo_db.events.delete_many({"run_id": run_id})

    def test_enqueue_returns_before_write_completes(self, mongo_db, tmp_path):
        """Enqueue is non-blocking: returns well before the MongoDB write finishes."""
        run_id = _unique_id()
        try:
            writer = BufferedWriter(backend=mongo_db, dlq_path=str(tmp_path / "dlq.jsonl"))
            start = time.monotonic()
            writer.create_run(run_id, "g", "t")
            elapsed = time.monotonic() - start
            writer.close()

            # enqueue must return in microseconds, not MongoDB-write time
            assert elapsed < 0.01
            assert mongo_db.runs.find_one({"_id": run_id}) is not None
        finally:
            mongo_db.runs.delete_one({"_id": run_id})
            mongo_db.events.delete_many({"run_id": run_id})

    def test_dropped_events_zero_on_clean_run(self, mongo_db, tmp_path):
        run_id = _unique_id()
        try:
            with BufferedWriter(backend=mongo_db, dlq_path=str(tmp_path / "dlq.jsonl")) as writer:
                writer.create_run(run_id, "g", "t")
                writer.complete_run(run_id)
            assert writer.dropped_events == 0
        finally:
            mongo_db.runs.delete_one({"_id": run_id})
            mongo_db.events.delete_many({"run_id": run_id})


# ---------------------------------------------------------------------------
# PostgreSQL — full lifecycle
# ---------------------------------------------------------------------------


def _fetch_run_pg(pg_db, run_id: str) -> dict | None:
    with pg_db._connection.cursor() as cur:
        cur.execute(
            """SELECT run_id, graph_id, thread_id, status, ended_at, error,
                      total_input_tokens, total_output_tokens, estimated_cost_usd
               FROM runs WHERE run_id = %s""",
            (run_id,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    keys = ["run_id", "graph_id", "thread_id", "status", "ended_at", "error",
            "total_input_tokens", "total_output_tokens", "estimated_cost_usd"]
    return dict(zip(keys, row))


def _fetch_events_pg(pg_db, run_id: str) -> list[dict]:
    with pg_db._connection.cursor() as cur:
        cur.execute(
            "SELECT event_type, node_name FROM events WHERE run_id = %s ORDER BY timestamp",
            (run_id,),
        )
        rows = cur.fetchall()
    return [{"event_type": r[0], "node_name": r[1]} for r in rows]


def _cleanup_pg(pg_db, run_id: str) -> None:
    with pg_db._connection.cursor() as cur:
        cur.execute("DELETE FROM events WHERE run_id = %s", (run_id,))
        cur.execute("DELETE FROM runs WHERE run_id = %s", (run_id,))


@pytest.mark.integration
class TestBufferedWriterPostgres:
    """BufferedWriter wrapping PostgresMonitorDB: writes must land in Postgres after close()."""

    def test_full_run_lifecycle_persists_to_postgres(self, pg_db, tmp_path):
        run_id = _unique_id()
        try:
            with BufferedWriter(backend=pg_db, dlq_path=str(tmp_path / "dlq.jsonl")) as writer:
                writer.create_run(run_id, "integration_graph", "thread-1")
                writer.insert_event(
                    run_id=run_id,
                    graph_id="integration_graph",
                    event_type="node_start",
                    node_name="agent",
                )
                writer.insert_event(
                    run_id=run_id,
                    graph_id="integration_graph",
                    event_type="node_end",
                    node_name="agent",
                    latency_ms=77.5,
                    input_tokens=20,
                    output_tokens=8,
                )
                writer.complete_run(run_id, total_input_tokens=20, total_output_tokens=8, estimated_cost_usd=0.0002)

            run_doc = _fetch_run_pg(pg_db, run_id)
            assert run_doc is not None
            assert run_doc["status"] == "completed"
            assert run_doc["graph_id"] == "integration_graph"
            assert run_doc["total_input_tokens"] == 20
            assert run_doc["total_output_tokens"] == 8

            events = _fetch_events_pg(pg_db, run_id)
            assert len(events) == 2
            assert events[0]["event_type"] == "node_start"
            assert events[1]["event_type"] == "node_end"
        finally:
            _cleanup_pg(pg_db, run_id)

    def test_failed_run_lifecycle_persists_to_postgres(self, pg_db, tmp_path):
        run_id = _unique_id()
        try:
            with BufferedWriter(backend=pg_db, dlq_path=str(tmp_path / "dlq.jsonl")) as writer:
                writer.create_run(run_id, "g", "t")
                writer.fail_run(run_id, "RuntimeError: something went wrong")

            run_doc = _fetch_run_pg(pg_db, run_id)
            assert run_doc["status"] == "failed"
            assert run_doc["error"] == "RuntimeError: something went wrong"
            assert run_doc["ended_at"] is not None
        finally:
            _cleanup_pg(pg_db, run_id)

    def test_concurrent_enqueues_all_land_in_postgres(self, pg_db, tmp_path):
        run_id = _unique_id()
        n_events = 30
        try:
            with BufferedWriter(backend=pg_db, dlq_path=str(tmp_path / "dlq.jsonl")) as writer:
                writer.create_run(run_id, "g", "t")
                for i in range(n_events):
                    writer.insert_event(
                        run_id=run_id,
                        graph_id="g",
                        event_type="node_start",
                        node_name=f"node_{i}",
                    )
                writer.complete_run(run_id)

            events = _fetch_events_pg(pg_db, run_id)
            assert len(events) == n_events
            assert writer.dropped_events == 0
        finally:
            _cleanup_pg(pg_db, run_id)


# ---------------------------------------------------------------------------
# DLQ — real backend that is intentionally unreachable
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestBufferedWriterDLQ:
    """DLQ is written when the backend raises on every attempt.

    Both backends (Mongo, Postgres) swallow connection errors internally and
    return silently — so the DLQ path is exercised via _FlakyBackend, which
    raises unconditionally and wraps the real backend connection.  This tests
    that BufferedWriter correctly routes exhausted writes to the DLQ file and
    that the data is absent from the live database.
    """

    def test_dlq_written_and_data_not_in_mongo(self, mongo_db, tmp_path):
        """Exhausted retries → DLQ entry written, data absent from Mongo."""
        run_id = _unique_id()
        dlq = tmp_path / "dlq.jsonl"
        # fail_count > max_retries so every write goes to DLQ
        flaky = _FlakyBackend(mongo_db, fail_count=10)
        try:
            with patch("stakeout_agent.writer.time.sleep"):
                with BufferedWriter(backend=flaky, max_retries=2, dlq_path=str(dlq)) as writer:
                    writer.create_run(run_id, "g", "t")
                    writer.insert_event(run_id=run_id, graph_id="g", event_type="node_start", node_name="n")

            assert mongo_db.runs.find_one({"_id": run_id}) is None, "run must not reach Mongo when all retries fail"
            assert dlq.exists()
            lines = dlq.read_text().strip().splitlines()
            assert len(lines) == 2
            methods = {json.loads(ln)["method"] for ln in lines}
            assert methods == {"create_run", "insert_event"}
            assert writer.dropped_events == 2
        finally:
            mongo_db.runs.delete_one({"_id": run_id})
            mongo_db.events.delete_many({"run_id": run_id})

    def test_dlq_written_and_data_not_in_postgres(self, pg_db, tmp_path):
        """Exhausted retries → DLQ entry written, data absent from Postgres."""
        run_id = _unique_id()
        dlq = tmp_path / "dlq.jsonl"
        flaky = _FlakyBackend(pg_db, fail_count=10)
        try:
            with patch("stakeout_agent.writer.time.sleep"):
                with BufferedWriter(backend=flaky, max_retries=2, dlq_path=str(dlq)) as writer:
                    writer.create_run(run_id, "g", "t")
                    writer.complete_run(run_id)

            assert _fetch_run_pg(pg_db, run_id) is None, "run must not reach Postgres when all retries fail"
            assert dlq.exists()
            lines = dlq.read_text().strip().splitlines()
            assert len(lines) == 2
            methods = {json.loads(ln)["method"] for ln in lines}
            assert methods == {"create_run", "complete_run"}
            assert writer.dropped_events == 2
        finally:
            _cleanup_pg(pg_db, run_id)


# ---------------------------------------------------------------------------
# Retry and recover — flaky wrapper around a real backend
# ---------------------------------------------------------------------------


class _FlakyBackend:
    """Delegates to a real backend but raises on the first `fail_count` calls per method."""

    def __init__(self, real, fail_count: int = 1):
        self._real = real
        self._fail_count = fail_count
        self._calls: dict[str, int] = {}

    def _maybe_fail(self, method: str) -> None:
        n = self._calls.get(method, 0) + 1
        self._calls[method] = n
        if n <= self._fail_count:
            raise OSError(f"injected transient failure #{n} for {method}")

    def create_run(self, *a, **kw):
        self._maybe_fail("create_run")
        return self._real.create_run(*a, **kw)

    def complete_run(self, *a, **kw):
        self._maybe_fail("complete_run")
        return self._real.complete_run(*a, **kw)

    def fail_run(self, *a, **kw):
        self._maybe_fail("fail_run")
        return self._real.fail_run(*a, **kw)

    def insert_event(self, *a, **kw):
        self._maybe_fail("insert_event")
        return self._real.insert_event(*a, **kw)


@pytest.mark.integration
class TestBufferedWriterRetryMongo:
    """BufferedWriter retries transient failures and the data still lands in Mongo."""

    def test_data_lands_after_transient_failure(self, mongo_db, tmp_path):
        run_id = _unique_id()
        flaky = _FlakyBackend(mongo_db, fail_count=1)
        try:
            with patch("stakeout_agent.writer.time.sleep"):  # skip backoff waits in test
                with BufferedWriter(
                    backend=flaky,
                    max_retries=3,
                    dlq_path=str(tmp_path / "dlq.jsonl"),
                ) as writer:
                    writer.create_run(run_id, "g", "t")
                    writer.complete_run(run_id)

            run_doc = mongo_db.runs.find_one({"_id": run_id})
            assert run_doc is not None
            assert run_doc["status"] == "completed"
            assert writer.dropped_events == 0
            # Each method was called twice: once failing, once succeeding
            assert flaky._calls["create_run"] == 2
            assert flaky._calls["complete_run"] == 2
        finally:
            mongo_db.runs.delete_one({"_id": run_id})
            mongo_db.events.delete_many({"run_id": run_id})

    def test_dlq_written_after_exhausted_retries_on_real_backend(self, mongo_db, tmp_path):
        """Writes that fail every attempt go to DLQ and never reach Mongo."""
        run_id = _unique_id()
        dlq = tmp_path / "dlq.jsonl"
        # Fail more times than max_retries so no write ever gets through
        flaky = _FlakyBackend(mongo_db, fail_count=10)
        try:
            with patch("stakeout_agent.writer.time.sleep"):
                with BufferedWriter(
                    backend=flaky,
                    max_retries=2,
                    dlq_path=str(dlq),
                ) as writer:
                    writer.create_run(run_id, "g", "t")

            assert mongo_db.runs.find_one({"_id": run_id}) is None
            assert dlq.exists()
            assert writer.dropped_events == 1
        finally:
            mongo_db.runs.delete_one({"_id": run_id})
            mongo_db.events.delete_many({"run_id": run_id})


@pytest.mark.integration
class TestBufferedWriterRetryPostgres:
    """BufferedWriter retries transient failures and the data still lands in Postgres."""

    def test_data_lands_after_transient_failure(self, pg_db, tmp_path):
        run_id = _unique_id()
        flaky = _FlakyBackend(pg_db, fail_count=1)
        try:
            with patch("stakeout_agent.writer.time.sleep"):
                with BufferedWriter(
                    backend=flaky,
                    max_retries=3,
                    dlq_path=str(tmp_path / "dlq.jsonl"),
                ) as writer:
                    writer.create_run(run_id, "g", "t")
                    writer.complete_run(run_id)

            run_doc = _fetch_run_pg(pg_db, run_id)
            assert run_doc is not None
            assert run_doc["status"] == "completed"
            assert writer.dropped_events == 0
            assert flaky._calls["create_run"] == 2
            assert flaky._calls["complete_run"] == 2
        finally:
            _cleanup_pg(pg_db, run_id)
