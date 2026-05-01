from __future__ import annotations

import threading
from contextlib import contextmanager
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from pymongo.errors import ConnectionFailure, OperationFailure

from stakeout_agent.backends.mongodb import MongoMonitorDB


def _make_mock_db():
    """Return (mock_db, mock_runs_col, mock_events_col) with update_one matched_count=1."""
    mock_runs = MagicMock()
    mock_events = MagicMock()

    result = MagicMock()
    result.matched_count = 1
    mock_runs.update_one.return_value = result

    mock_db = MagicMock()
    mock_db.runs = mock_runs
    mock_db.events = mock_events
    return mock_db, mock_runs, mock_events


@contextmanager
def _patched_monitor(mock_db):
    with patch("stakeout_agent.backends.mongodb._make_client", return_value=mock_db):
        yield MongoMonitorDB()


# ---------------------------------------------------------------------------
# create_run
# ---------------------------------------------------------------------------


class TestCreateRun:
    def test_write_error_is_logged_not_raised(self, caplog):
        mock_db, mock_runs, _ = _make_mock_db()
        mock_runs.insert_one.side_effect = OperationFailure("disk full")
        with _patched_monitor(mock_db) as monitor:
            monitor.create_run("run-1", "g", "t")  # must not raise

        assert any("create_run" in r.message and "run-1" in r.message for r in caplog.records)

    def test_inserts_correct_document(self):
        mock_db, mock_runs, _ = _make_mock_db()
        with _patched_monitor(mock_db) as monitor:
            monitor.create_run("run-1", "my_graph", "thread-42")

        mock_runs.insert_one.assert_called_once()
        doc = mock_runs.insert_one.call_args.args[0]
        assert doc["_id"] == "run-1"
        assert doc["graph_id"] == "my_graph"
        assert doc["thread_id"] == "thread-42"
        assert doc["status"] == "running"
        assert isinstance(doc["started_at"], datetime)
        assert doc["ended_at"] is None
        assert doc["error"] is None
        assert doc["metadata"] == {}


# ---------------------------------------------------------------------------
# complete_run
# ---------------------------------------------------------------------------


class TestCompleteRun:
    def test_sets_completed_status(self):
        mock_db, mock_runs, _ = _make_mock_db()
        with _patched_monitor(mock_db) as monitor:
            monitor.complete_run("run-1")

        mock_runs.update_one.assert_called_once()
        filt, update = mock_runs.update_one.call_args.args
        assert filt == {"_id": "run-1"}
        assert update["$set"]["status"] == "completed"
        assert isinstance(update["$set"]["ended_at"], datetime)

    def test_write_error_is_logged_not_raised(self, caplog):
        mock_db, mock_runs, _ = _make_mock_db()
        mock_runs.update_one.side_effect = OperationFailure("timeout")
        with _patched_monitor(mock_db) as monitor:
            monitor.complete_run("run-1")  # must not raise

        assert any("complete_run" in r.message and "run-1" in r.message for r in caplog.records)

    def test_missing_run_id_logs_warning(self, caplog):
        mock_db, mock_runs, _ = _make_mock_db()
        mock_runs.update_one.return_value.matched_count = 0
        with _patched_monitor(mock_db) as monitor:
            monitor.complete_run("nonexistent")  # must not raise

        assert any("nonexistent" in r.message for r in caplog.records)

    def test_missing_run_id_does_not_raise(self):
        mock_db, mock_runs, _ = _make_mock_db()
        mock_runs.update_one.return_value.matched_count = 0
        with _patched_monitor(mock_db) as monitor:
            monitor.complete_run("nonexistent")  # must not raise


# ---------------------------------------------------------------------------
# fail_run
# ---------------------------------------------------------------------------


class TestFailRun:
    def test_sets_failed_status_and_error(self):
        mock_db, mock_runs, _ = _make_mock_db()
        with _patched_monitor(mock_db) as monitor:
            monitor.fail_run("run-1", "boom")

        mock_runs.update_one.assert_called_once()
        _, update = mock_runs.update_one.call_args.args
        assert update["$set"]["status"] == "failed"
        assert update["$set"]["error"] == "boom"
        assert isinstance(update["$set"]["ended_at"], datetime)

    def test_write_error_is_logged_not_raised(self, caplog):
        mock_db, mock_runs, _ = _make_mock_db()
        mock_runs.update_one.side_effect = OperationFailure("timeout")
        with _patched_monitor(mock_db) as monitor:
            monitor.fail_run("run-1", "boom")  # must not raise

        assert any("fail_run" in r.message and "run-1" in r.message for r in caplog.records)

    def test_missing_run_id_logs_warning(self, caplog):
        mock_db, mock_runs, _ = _make_mock_db()
        mock_runs.update_one.return_value.matched_count = 0
        with _patched_monitor(mock_db) as monitor:
            monitor.fail_run("nonexistent", "error")  # must not raise

        assert any("nonexistent" in r.message for r in caplog.records)

    def test_missing_run_id_does_not_raise(self):
        mock_db, mock_runs, _ = _make_mock_db()
        mock_runs.update_one.return_value.matched_count = 0
        with _patched_monitor(mock_db) as monitor:
            monitor.fail_run("nonexistent", "error")  # must not raise


# ---------------------------------------------------------------------------
# insert_event
# ---------------------------------------------------------------------------


class TestInsertEvent:
    def test_inserts_correct_document(self):
        mock_db, _, mock_events = _make_mock_db()
        with _patched_monitor(mock_db) as monitor:
            monitor.insert_event(
                run_id="run-1",
                graph_id="g",
                event_type="node_start",
                node_name="my_node",
                latency_ms=12.5,
                payload={"x": 1},
                error=None,
            )

        mock_events.insert_one.assert_called_once()
        doc = mock_events.insert_one.call_args.args[0]
        assert doc["run_id"] == "run-1"
        assert doc["graph_id"] == "g"
        assert doc["event_type"] == "node_start"
        assert doc["node_name"] == "my_node"
        assert isinstance(doc["timestamp"], datetime)
        assert doc["latency_ms"] == 12.5
        assert doc["payload"] == {"x": 1}
        assert doc["error"] is None

    def test_write_error_is_logged_not_raised(self, caplog):
        mock_db, _, mock_events = _make_mock_db()
        mock_events.insert_one.side_effect = OperationFailure("disk full")
        with _patched_monitor(mock_db) as monitor:
            monitor.insert_event(run_id="run-1", graph_id="g", event_type="e", node_name="n")  # must not raise

        assert any("insert_event" in r.message and "run-1" in r.message for r in caplog.records)

    def test_payload_defaults_to_empty_dict(self):
        mock_db, _, mock_events = _make_mock_db()
        with _patched_monitor(mock_db) as monitor:
            monitor.insert_event(run_id="r", graph_id="g", event_type="e", node_name="n")

        doc = mock_events.insert_one.call_args.args[0]
        assert doc["payload"] == {}


# ---------------------------------------------------------------------------
# Connection failure
# ---------------------------------------------------------------------------


class TestConnectionFailure:
    def test_propagates_on_first_operation(self):
        with patch("stakeout_agent.backends.mongodb._make_client", side_effect=ConnectionFailure("refused")):
            monitor = MongoMonitorDB()
            with pytest.raises(ConnectionFailure):
                monitor.create_run("r", "g", "t")


# ---------------------------------------------------------------------------
# Index creation
# ---------------------------------------------------------------------------


class TestIndexCreation:
    def test_creates_all_five_indexes(self):
        from pymongo import DESCENDING

        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_client.__getitem__ = MagicMock(return_value=mock_db)

        with patch("stakeout_agent.backends.mongodb.MongoClient", return_value=mock_client):
            monitor = MongoMonitorDB()
            _ = monitor._conn  # trigger lazy init

        create_index_calls = mock_db.runs.create_index.call_args_list + mock_db.events.create_index.call_args_list
        assert len(create_index_calls) == 5

        index_args = [c.args[0] for c in create_index_calls]
        assert [("started_at", DESCENDING)] in index_args
        assert "graph_id" in index_args
        assert "status" in index_args
        assert "run_id" in index_args
        assert [("timestamp", DESCENDING)] in index_args


# ---------------------------------------------------------------------------
# Concurrent access to _conn
# ---------------------------------------------------------------------------


class TestConcurrentConn:
    def test_make_client_called_once_under_concurrent_access(self):
        mock_db, _, _ = _make_mock_db()
        call_count = 0

        def slow_make_client():
            nonlocal call_count
            call_count += 1
            return mock_db

        n_threads = 20
        barrier = threading.Barrier(n_threads)
        monitor = MongoMonitorDB()

        def access():
            barrier.wait()
            _ = monitor._conn

        with patch("stakeout_agent.backends.mongodb._make_client", side_effect=slow_make_client):
            threads = [threading.Thread(target=access) for _ in range(n_threads)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert call_count == 1
