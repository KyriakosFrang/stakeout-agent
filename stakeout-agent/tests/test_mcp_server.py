"""Unit tests for the MCP server tools and resources.

The mcp package is required; these tests are skipped when it is not installed.
All database calls use a MagicMock that implements AbstractQueryDB.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

pytest.importorskip("mcp", reason="mcp package not installed")

from stakeout_agent.mcp.server import create_server  # noqa: E402, I001


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_db():
    db = MagicMock()
    return db


def _make_run(
    run_id: str = "run-1",
    graph_id: str = "my-graph",
    status: str = "completed",
    latency_ms: float | None = 500.0,
    cost: float | None = 0.05,
    error: str | None = None,
) -> dict:
    return {
        "run_id": run_id,
        "graph_id": graph_id,
        "thread_id": "t1",
        "status": status,
        "started_at": "2026-01-01T00:00:00+00:00",
        "ended_at": "2026-01-01T00:00:00.500000+00:00",
        "error": error,
        "latency_ms": latency_ms,
        "estimated_cost_usd": cost,
        "total_input_tokens": 100,
        "total_output_tokens": 50,
    }


def _call_tool(server, tool_name: str, **kwargs):
    """Directly invoke a registered tool function by name."""
    import asyncio

    tool = server._tool_manager._tools[tool_name]
    result = tool.fn(**kwargs)
    if asyncio.iscoroutine(result):
        return asyncio.get_event_loop().run_until_complete(result)
    return result


# ---------------------------------------------------------------------------
# get_recent_runs
# ---------------------------------------------------------------------------


class TestGetRecentRuns:
    def test_delegates_to_db(self):
        db = _mock_db()
        db.query_recent_runs.return_value = [_make_run()]
        server = create_server(db)
        _call_tool(server, "get_recent_runs", graph_id="my-graph", limit=5)
        db.query_recent_runs.assert_called_once_with("my-graph", 5)

    def test_empty_graph_id_becomes_none(self):
        db = _mock_db()
        db.query_recent_runs.return_value = []
        server = create_server(db)
        _call_tool(server, "get_recent_runs", graph_id="", limit=10)
        db.query_recent_runs.assert_called_once_with(None, 10)

    def test_limit_capped_at_100(self):
        db = _mock_db()
        db.query_recent_runs.return_value = []
        server = create_server(db)
        _call_tool(server, "get_recent_runs", graph_id="", limit=999)
        db.query_recent_runs.assert_called_once_with(None, 100)

    def test_limit_floor_at_1(self):
        db = _mock_db()
        db.query_recent_runs.return_value = []
        server = create_server(db)
        _call_tool(server, "get_recent_runs", graph_id="", limit=0)
        db.query_recent_runs.assert_called_once_with(None, 1)

    def test_returns_db_result(self):
        db = _mock_db()
        expected = [_make_run("r1"), _make_run("r2")]
        db.query_recent_runs.return_value = expected
        server = create_server(db)
        result = _call_tool(server, "get_recent_runs", graph_id="g", limit=2)
        assert result == expected


# ---------------------------------------------------------------------------
# get_run_detail
# ---------------------------------------------------------------------------


class TestGetRunDetail:
    def test_found(self):
        db = _mock_db()
        run = _make_run("r1")
        db.query_run_detail.return_value = {"run": run, "events": []}
        server = create_server(db)
        result = _call_tool(server, "get_run_detail", run_id="r1")
        assert result["run"]["run_id"] == "r1"
        db.query_run_detail.assert_called_once_with("r1")

    def test_not_found_returns_error_dict(self):
        db = _mock_db()
        db.query_run_detail.return_value = None
        server = create_server(db)
        result = _call_tool(server, "get_run_detail", run_id="missing")
        assert "error" in result


# ---------------------------------------------------------------------------
# get_failed_runs
# ---------------------------------------------------------------------------


class TestGetFailedRuns:
    def test_delegates_time_window(self):
        db = _mock_db()
        db.query_failed_runs.return_value = []
        server = create_server(db)
        before = time.time()
        _call_tool(server, "get_failed_runs", graph_id="g", last_n_hours=2.0)
        after = time.time()
        db.query_failed_runs.assert_called_once()
        _, since_ts = db.query_failed_runs.call_args[0]
        assert before - 2 * 3600 - 1 <= since_ts <= after - 2 * 3600 + 1

    def test_returns_failed_runs(self):
        db = _mock_db()
        failed = [_make_run("r1", status="failed", error="timeout")]
        db.query_failed_runs.return_value = failed
        server = create_server(db)
        result = _call_tool(server, "get_failed_runs", graph_id="g", last_n_hours=1.0)
        assert result == failed


# ---------------------------------------------------------------------------
# get_slow_runs
# ---------------------------------------------------------------------------


class TestGetSlowRuns:
    def test_delegates_threshold_and_window(self):
        db = _mock_db()
        db.query_slow_runs.return_value = []
        server = create_server(db)
        _call_tool(server, "get_slow_runs", threshold_ms=5000.0, graph_id="g", last_n_hours=12.0)
        db.query_slow_runs.assert_called_once()
        g_arg, thresh_arg, since_arg = db.query_slow_runs.call_args[0]
        assert g_arg == "g"
        assert thresh_arg == 5000.0
        assert since_arg == pytest.approx(time.time() - 12 * 3600, abs=2)


# ---------------------------------------------------------------------------
# get_run_stats
# ---------------------------------------------------------------------------


class TestGetRunStats:
    def test_delegates_days_window(self):
        db = _mock_db()
        db.query_run_stats.return_value = {"run_count": 0, "error_rate": None}
        server = create_server(db)
        _call_tool(server, "get_run_stats", graph_id="g", last_n_days=3.0)
        db.query_run_stats.assert_called_once()
        g_arg, since_arg = db.query_run_stats.call_args[0]
        assert g_arg == "g"
        assert since_arg == pytest.approx(time.time() - 3 * 86400, abs=2)

    def test_returns_stats(self):
        db = _mock_db()
        stats = {"run_count": 10, "error_rate": 0.1, "p50_latency_ms": 300.0, "p95_latency_ms": 800.0}
        db.query_run_stats.return_value = stats
        server = create_server(db)
        result = _call_tool(server, "get_run_stats", graph_id="", last_n_days=7.0)
        assert result == stats


# ---------------------------------------------------------------------------
# search_runs_by_output
# ---------------------------------------------------------------------------


class TestSearchRunsByOutput:
    def test_delegates_query(self):
        db = _mock_db()
        db.query_runs_by_output.return_value = []
        server = create_server(db)
        _call_tool(server, "search_runs_by_output", query="rate limit", graph_id="g", limit=5)
        db.query_runs_by_output.assert_called_once_with("g", "rate limit", 5)

    def test_empty_graph_id_becomes_none(self):
        db = _mock_db()
        db.query_runs_by_output.return_value = []
        server = create_server(db)
        _call_tool(server, "search_runs_by_output", query="err", graph_id="", limit=10)
        db.query_runs_by_output.assert_called_once_with(None, "err", 10)


# ---------------------------------------------------------------------------
# AbstractQueryDB implementation parity: Mongo helpers
# ---------------------------------------------------------------------------


class TestMongoHelpers:
    """Verify the Mongo serialisation helpers produce correct output."""

    def test_ser_run_renames_id(self):
        from datetime import datetime, timezone

        from stakeout_agent.backends.mongodb import _ser_run

        doc = {
            "_id": "run-abc",
            "graph_id": "g",
            "thread_id": "t",
            "status": "completed",
            "started_at": datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            "ended_at": datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
            "error": None,
        }
        result = _ser_run(doc)
        assert result["run_id"] == "run-abc"
        assert "run_id" in result
        assert "_id" not in result
        assert result["latency_ms"] == pytest.approx(1000.0)
        assert isinstance(result["started_at"], str)

    def test_ser_run_null_ended_at(self):
        from datetime import datetime, timezone

        from stakeout_agent.backends.mongodb import _ser_run

        doc = {
            "_id": "r",
            "graph_id": "g",
            "thread_id": "t",
            "status": "running",
            "started_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
            "ended_at": None,
            "error": None,
        }
        result = _ser_run(doc)
        assert result["latency_ms"] is None

    def test_percentile_single_value(self):
        from stakeout_agent.backends.mongodb import _percentile

        assert _percentile([500.0], 95) == pytest.approx(500.0)

    def test_percentile_empty(self):
        from stakeout_agent.backends.mongodb import _percentile

        assert _percentile([], 95) is None

    def test_percentile_multiple(self):
        from stakeout_agent.backends.mongodb import _percentile

        values = [float(i) for i in range(1, 101)]
        result = _percentile(values, 95)
        assert result is not None
        assert result > 90.0


# ---------------------------------------------------------------------------
# AbstractQueryDB implementation parity: Postgres helpers
# ---------------------------------------------------------------------------


class TestPostgresHelpers:
    def test_row_to_run_computes_latency(self):
        from datetime import datetime, timezone

        from stakeout_agent.backends.postgres import _row_to_run

        cols = ("run_id", "graph_id", "thread_id", "status", "started_at", "ended_at", "error",
                "run_inputs", "parent_run_id", "prompt_version",
                "total_input_tokens", "total_output_tokens", "estimated_cost_usd",
                "total_cache_read_tokens", "total_cache_creation_tokens")
        row = (
            "r1", "g", "t", "completed",
            datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 1, 1, 0, 0, 2, tzinfo=timezone.utc),
            None, None, None, None, None, None, None, None, None,
        )
        result = _row_to_run(cols, row)
        assert result["run_id"] == "r1"
        assert result["latency_ms"] == pytest.approx(2000.0)
        assert isinstance(result["started_at"], str)

    def test_row_to_run_none_ended_at(self):
        from datetime import datetime, timezone

        from stakeout_agent.backends.postgres import _row_to_run

        cols = ("run_id", "graph_id", "thread_id", "status", "started_at", "ended_at", "error",
                "run_inputs", "parent_run_id", "prompt_version",
                "total_input_tokens", "total_output_tokens", "estimated_cost_usd",
                "total_cache_read_tokens", "total_cache_creation_tokens")
        row = (
            "r2", "g", "t", "running",
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            None,
            None, None, None, None, None, None, None, None, None,
        )
        result = _row_to_run(cols, row)
        assert result["latency_ms"] is None
