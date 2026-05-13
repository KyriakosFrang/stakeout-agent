"""Integration tests for PostgresMonitorDB against a real PostgreSQL instance.

Requires: docker compose up -d postgres
Skipped automatically when PostgreSQL is not reachable.
"""

from __future__ import annotations

import pytest


def _fetch_run(pg_db, run_id: str) -> dict | None:
    with pg_db._connection.cursor() as cur:
        cur.execute(
            """SELECT run_id, graph_id, thread_id, status, ended_at, error,
                      total_input_tokens, total_output_tokens, estimated_cost_usd,
                      total_cache_read_tokens, total_cache_creation_tokens
               FROM runs WHERE run_id = %s""",
            (run_id,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    keys = [
        "run_id",
        "graph_id",
        "thread_id",
        "status",
        "ended_at",
        "error",
        "total_input_tokens",
        "total_output_tokens",
        "estimated_cost_usd",
        "total_cache_read_tokens",
        "total_cache_creation_tokens",
    ]
    return dict(zip(keys, row))


def _fetch_events(pg_db, run_id: str) -> list[dict]:
    with pg_db._connection.cursor() as cur:
        cur.execute(
            """SELECT event_type, node_name, latency_ms, error,
                      input_tokens, output_tokens, model,
                      cache_read_tokens, cache_creation_tokens
               FROM events WHERE run_id = %s ORDER BY timestamp""",
            (run_id,),
        )
        rows = cur.fetchall()
    keys = [
        "event_type",
        "node_name",
        "latency_ms",
        "error",
        "input_tokens",
        "output_tokens",
        "model",
        "cache_read_tokens",
        "cache_creation_tokens",
    ]
    return [dict(zip(keys, row)) for row in rows]


def _cleanup(pg_db, run_id: str) -> None:
    with pg_db._connection.cursor() as cur:
        cur.execute("DELETE FROM events WHERE run_id = %s", (run_id,))
        cur.execute("DELETE FROM runs WHERE run_id = %s", (run_id,))


@pytest.mark.integration
class TestPostgresCreateRun:
    def test_run_document_created_with_running_status(self, pg_db, run_id):
        try:
            pg_db.create_run(run_id, "test_graph", "test_thread")

            doc = _fetch_run(pg_db, run_id)
            assert doc is not None
            assert doc["status"] == "running"
            assert doc["graph_id"] == "test_graph"
            assert doc["thread_id"] == "test_thread"
            assert doc["ended_at"] is None
            assert doc["error"] is None
        finally:
            _cleanup(pg_db, run_id)


@pytest.mark.integration
class TestPostgresCompleteRun:
    def test_run_marked_completed_with_token_fields(self, pg_db, run_id):
        try:
            pg_db.create_run(run_id, "g", "t")
            pg_db.complete_run(
                run_id,
                total_input_tokens=120,
                total_output_tokens=40,
                estimated_cost_usd=0.001,
                total_cache_read_tokens=10,
                total_cache_creation_tokens=5,
            )

            doc = _fetch_run(pg_db, run_id)
            assert doc["status"] == "completed"
            assert doc["ended_at"] is not None
            assert doc["total_input_tokens"] == 120
            assert doc["total_output_tokens"] == 40
            assert doc["estimated_cost_usd"] == pytest.approx(0.001)
            assert doc["total_cache_read_tokens"] == 10
            assert doc["total_cache_creation_tokens"] == 5
        finally:
            _cleanup(pg_db, run_id)


@pytest.mark.integration
class TestPostgresFailRun:
    def test_run_marked_failed_with_error(self, pg_db, run_id):
        try:
            pg_db.create_run(run_id, "g", "t")
            pg_db.fail_run(run_id, "ValueError: something broke")

            doc = _fetch_run(pg_db, run_id)
            assert doc["status"] == "failed"
            assert doc["ended_at"] is not None
            assert doc["error"] == "ValueError: something broke"
        finally:
            _cleanup(pg_db, run_id)


@pytest.mark.integration
class TestPostgresInsertEvent:
    def test_node_start_event_stored(self, pg_db, run_id):
        try:
            pg_db.create_run(run_id, "g", "t")
            pg_db.insert_event(
                run_id=run_id,
                graph_id="g",
                event_type="node_start",
                node_name="agent",
            )

            events = _fetch_events(pg_db, run_id)
            assert len(events) == 1
            assert events[0]["event_type"] == "node_start"
            assert events[0]["node_name"] == "agent"
        finally:
            _cleanup(pg_db, run_id)

    def test_node_end_event_stores_tokens_and_model(self, pg_db, run_id):
        try:
            pg_db.create_run(run_id, "g", "t")
            pg_db.insert_event(
                run_id=run_id,
                graph_id="g",
                event_type="node_end",
                node_name="agent",
                latency_ms=250.5,
                input_tokens=80,
                output_tokens=30,
                model="gpt-4o",
                cache_read_tokens=20,
                cache_creation_tokens=0,
            )

            events = _fetch_events(pg_db, run_id)
            assert len(events) == 1
            ev = events[0]
            assert ev["latency_ms"] == pytest.approx(250.5)
            assert ev["input_tokens"] == 80
            assert ev["output_tokens"] == 30
            assert ev["model"] == "gpt-4o"
            assert ev["cache_read_tokens"] == 20
        finally:
            _cleanup(pg_db, run_id)

    def test_error_event_stored_with_error_field(self, pg_db, run_id):
        try:
            pg_db.create_run(run_id, "g", "t")
            pg_db.insert_event(
                run_id=run_id,
                graph_id="g",
                event_type="error",
                node_name="agent",
                error="TimeoutError: request timed out",
            )

            events = _fetch_events(pg_db, run_id)
            assert len(events) == 1
            assert events[0]["error"] == "TimeoutError: request timed out"
        finally:
            _cleanup(pg_db, run_id)


@pytest.mark.integration
class TestPostgresFullLifecycle:
    def test_complete_run_lifecycle_event_count(self, pg_db, run_id):
        try:
            pg_db.create_run(run_id, "my_graph", "thread-1")
            pg_db.insert_event(run_id=run_id, graph_id="my_graph", event_type="node_start", node_name="agent")
            pg_db.insert_event(
                run_id=run_id,
                graph_id="my_graph",
                event_type="node_end",
                node_name="agent",
                latency_ms=100.0,
                input_tokens=50,
                output_tokens=20,
            )
            pg_db.insert_event(run_id=run_id, graph_id="my_graph", event_type="tool_call", node_name="search")
            pg_db.insert_event(
                run_id=run_id, graph_id="my_graph", event_type="tool_result", node_name="search", latency_ms=30.0
            )
            pg_db.complete_run(run_id, total_input_tokens=50, total_output_tokens=20)

            doc = _fetch_run(pg_db, run_id)
            assert doc["status"] == "completed"

            events = _fetch_events(pg_db, run_id)
            assert len(events) == 4
        finally:
            _cleanup(pg_db, run_id)
