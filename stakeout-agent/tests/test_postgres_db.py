from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from stakeout_agent.backends.postgres import PostgresMonitorDB


def _make_mock_conn(rowcount: int = 1):
    """Return (mock_conn, mock_cursor) wired for use as context manager."""
    mock_cursor = MagicMock()
    mock_cursor.rowcount = rowcount

    # Make cursor() work as a context manager
    mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
    mock_cursor.__exit__ = MagicMock(return_value=False)

    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    return mock_conn, mock_cursor


@contextmanager
def _patched_postgres(mock_conn):
    with patch("stakeout_agent.backends.postgres._make_pg_conn", return_value=mock_conn):
        yield PostgresMonitorDB()


# ---------------------------------------------------------------------------
# create_run
# ---------------------------------------------------------------------------


class TestCreateRun:
    def test_inserts_correct_sql(self):
        mock_conn, mock_cursor = _make_mock_conn()
        with _patched_postgres(mock_conn) as pg:
            pg.create_run("run-1", "my_graph", "thread-42")

        mock_cursor.execute.assert_called_once()
        sql, params = mock_cursor.execute.call_args.args
        assert "INSERT INTO runs" in sql
        assert params[0] == "run-1"
        assert params[1] == "my_graph"
        assert params[2] == "thread-42"
        assert isinstance(params[3], datetime)

    def test_write_error_is_logged_not_raised(self, caplog):
        mock_conn, mock_cursor = _make_mock_conn()
        mock_cursor.execute.side_effect = Exception("disk full")
        with _patched_postgres(mock_conn) as pg:
            pg.create_run("run-1", "g", "t")  # must not raise

        assert any("create_run" in r.message and "run-1" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# complete_run
# ---------------------------------------------------------------------------


class TestCompleteRun:
    def test_sets_completed_status(self):
        mock_conn, mock_cursor = _make_mock_conn()
        with _patched_postgres(mock_conn) as pg:
            pg.complete_run("run-1")

        mock_cursor.execute.assert_called_once()
        sql, params = mock_cursor.execute.call_args.args
        assert "completed" in sql
        assert isinstance(params[0], datetime)
        assert params[1] == "run-1"

    def test_write_error_is_logged_not_raised(self, caplog):
        mock_conn, mock_cursor = _make_mock_conn()
        mock_cursor.execute.side_effect = Exception("timeout")
        with _patched_postgres(mock_conn) as pg:
            pg.complete_run("run-1")  # must not raise

        assert any("complete_run" in r.message and "run-1" in r.message for r in caplog.records)

    def test_missing_run_id_logs_warning(self, caplog):
        mock_conn, mock_cursor = _make_mock_conn(rowcount=0)
        with _patched_postgres(mock_conn) as pg:
            pg.complete_run("nonexistent")

        assert any("nonexistent" in r.message for r in caplog.records)

    def test_missing_run_id_does_not_raise(self):
        mock_conn, mock_cursor = _make_mock_conn(rowcount=0)
        with _patched_postgres(mock_conn) as pg:
            pg.complete_run("nonexistent")  # must not raise


# ---------------------------------------------------------------------------
# fail_run
# ---------------------------------------------------------------------------


class TestFailRun:
    def test_sets_failed_status_and_error(self):
        mock_conn, mock_cursor = _make_mock_conn()
        with _patched_postgres(mock_conn) as pg:
            pg.fail_run("run-1", "boom")

        mock_cursor.execute.assert_called_once()
        sql, params = mock_cursor.execute.call_args.args
        assert "failed" in sql
        assert isinstance(params[0], datetime)
        assert params[1] == "boom"
        assert params[2] == "run-1"

    def test_write_error_is_logged_not_raised(self, caplog):
        mock_conn, mock_cursor = _make_mock_conn()
        mock_cursor.execute.side_effect = Exception("timeout")
        with _patched_postgres(mock_conn) as pg:
            pg.fail_run("run-1", "boom")  # must not raise

        assert any("fail_run" in r.message and "run-1" in r.message for r in caplog.records)

    def test_missing_run_id_logs_warning(self, caplog):
        mock_conn, mock_cursor = _make_mock_conn(rowcount=0)
        with _patched_postgres(mock_conn) as pg:
            pg.fail_run("nonexistent", "error")

        assert any("nonexistent" in r.message for r in caplog.records)

    def test_missing_run_id_does_not_raise(self):
        mock_conn, mock_cursor = _make_mock_conn(rowcount=0)
        with _patched_postgres(mock_conn) as pg:
            pg.fail_run("nonexistent", "error")  # must not raise


# ---------------------------------------------------------------------------
# insert_event
# ---------------------------------------------------------------------------


class TestInsertEvent:
    def test_inserts_correct_sql(self):
        mock_conn, mock_cursor = _make_mock_conn()
        with _patched_postgres(mock_conn) as pg:
            pg.insert_event(
                run_id="run-1",
                graph_id="g",
                event_type="node_start",
                node_name="my_node",
                latency_ms=12.5,
                payload={"x": 1},
                error=None,
            )

        mock_cursor.execute.assert_called_once()
        sql, params = mock_cursor.execute.call_args.args
        assert "INSERT INTO events" in sql
        assert params[0] == "run-1"
        assert params[1] == "g"
        assert params[2] == "node_start"
        assert params[3] == "my_node"
        assert params[4] == 12.5
        assert json.loads(params[5]) == {"x": 1}
        assert params[6] is None
        assert params[7] is None  # messages not provided
        assert isinstance(params[8], datetime)

    def test_payload_defaults_to_empty_dict(self):
        mock_conn, mock_cursor = _make_mock_conn()
        with _patched_postgres(mock_conn) as pg:
            pg.insert_event(run_id="r", graph_id="g", event_type="e", node_name="n")

        _, params = mock_cursor.execute.call_args.args
        assert json.loads(params[5]) == {}

    def test_messages_serialised_to_json(self):
        mock_conn, mock_cursor = _make_mock_conn()
        msgs = [{"role": "human", "content": "hello"}]
        with _patched_postgres(mock_conn) as pg:
            pg.insert_event(run_id="r", graph_id="g", event_type="e", node_name="n", messages=msgs)

        _, params = mock_cursor.execute.call_args.args
        assert json.loads(params[7]) == msgs

    def test_write_error_is_logged_not_raised(self, caplog):
        mock_conn, mock_cursor = _make_mock_conn()
        mock_cursor.execute.side_effect = Exception("disk full")
        with _patched_postgres(mock_conn) as pg:
            pg.insert_event(run_id="run-1", graph_id="g", event_type="e", node_name="n")  # must not raise

        assert any("insert_event" in r.message and "run-1" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Connection failure
# ---------------------------------------------------------------------------


class TestConnectionFailure:
    def test_propagates_on_first_operation(self):
        with patch("stakeout_agent.backends.postgres._make_pg_conn", side_effect=Exception("refused")):
            pg = PostgresMonitorDB()
            with pytest.raises(Exception, match="refused"):
                pg.create_run("r", "g", "t")


# ---------------------------------------------------------------------------
# Table / index creation
# ---------------------------------------------------------------------------


class TestTableCreation:
    def test_connection_is_established_lazily(self):
        mock_conn, mock_cursor = _make_mock_conn()
        pg = PostgresMonitorDB()
        assert pg._conn is None  # not connected yet
        with patch("stakeout_agent.backends.postgres._make_pg_conn", return_value=mock_conn):
            conn = pg._connection
        assert conn is mock_conn

    def test_sql_contains_required_tables(self):
        from stakeout_agent.backends.postgres import _CREATE_TABLES_SQL

        assert "CREATE TABLE IF NOT EXISTS runs" in _CREATE_TABLES_SQL
        assert "CREATE TABLE IF NOT EXISTS events" in _CREATE_TABLES_SQL
        assert "idx_runs_started_at" in _CREATE_TABLES_SQL
        assert "idx_runs_graph_id" in _CREATE_TABLES_SQL
        assert "idx_runs_status" in _CREATE_TABLES_SQL
        assert "idx_events_run_id" in _CREATE_TABLES_SQL
        assert "idx_events_timestamp" in _CREATE_TABLES_SQL


# ---------------------------------------------------------------------------
# Concurrent access to _connection
# ---------------------------------------------------------------------------


class TestConcurrentConn:
    def test_make_pg_conn_called_once_under_concurrent_access(self):
        mock_conn, _ = _make_mock_conn()
        call_count = 0

        def slow_make_conn():
            nonlocal call_count
            call_count += 1
            return mock_conn

        n_threads = 20
        barrier = threading.Barrier(n_threads)
        pg = PostgresMonitorDB()

        def access():
            barrier.wait()
            _ = pg._connection

        with patch("stakeout_agent.backends.postgres._make_pg_conn", side_effect=slow_make_conn):
            threads = [threading.Thread(target=access) for _ in range(n_threads)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert call_count == 1


# ---------------------------------------------------------------------------
# get_backend factory
# ---------------------------------------------------------------------------


class TestGetBackend:
    def test_returns_mongo_by_default(self, monkeypatch):
        monkeypatch.delenv("STAKEOUT_BACKEND", raising=False)
        from stakeout_agent.backends import get_backend
        from stakeout_agent.backends.mongodb import MongoMonitorDB

        result = get_backend()
        assert isinstance(result, MongoMonitorDB)

    def test_returns_postgres_when_configured(self, monkeypatch):
        monkeypatch.setenv("STAKEOUT_BACKEND", "postgres")
        from stakeout_agent.backends import get_backend

        result = get_backend()
        assert isinstance(result, PostgresMonitorDB)

    def test_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("STAKEOUT_BACKEND", "POSTGRES")
        from stakeout_agent.backends import get_backend

        result = get_backend()
        assert isinstance(result, PostgresMonitorDB)
