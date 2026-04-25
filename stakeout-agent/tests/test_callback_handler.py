from __future__ import annotations

import time
from unittest.mock import MagicMock
from uuid import UUID, uuid4

from stakeout_agent.callback_handler import AsyncLangGraphMonitorCallback, LangGraphMonitorCallback, _MonitorBase


def make_uuid() -> UUID:
    return uuid4()


def mock_db() -> MagicMock:
    return MagicMock()


# ---------------------------------------------------------------------------
# _MonitorBase helpers
# ---------------------------------------------------------------------------


class TestExtractName:
    def test_from_serialized_name(self):
        assert _MonitorBase._extract_name({"name": "my_node"}, {}) == "my_node"

    def test_from_serialized_id(self):
        assert _MonitorBase._extract_name({"id": ["pkg", "MyNode"]}, {}) == "MyNode"

    def test_from_kwargs_when_no_serialized(self):
        assert _MonitorBase._extract_name(None, {"name": "fallback"}) == "fallback"

    def test_unknown_when_nothing_available(self):
        assert _MonitorBase._extract_name(None, {}) == "unknown"


class TestPopLatency:
    def test_returns_positive_milliseconds(self):
        store = {"k": time.monotonic() - 0.1}
        latency = _MonitorBase._pop_latency(store, "k")
        assert latency is not None
        assert latency >= 100.0
        assert "k" not in store

    def test_missing_key_returns_none(self):
        assert _MonitorBase._pop_latency({}, "missing") is None


class TestSafeTruncate:
    def test_short_data_returned_unchanged(self):
        assert _MonitorBase._safe_truncate("hello") == "hello"

    def test_long_string_truncated(self):
        long_str = "x" * 600
        result = _MonitorBase._safe_truncate(long_str)
        assert result == "x" * 500

    def test_non_string_returned_when_short(self):
        data = {"key": "value"}
        assert _MonitorBase._safe_truncate(data) == data

    def test_unserializable_returns_empty_dict(self):
        class Bad:
            def __str__(self):
                raise RuntimeError("boom")

        assert _MonitorBase._safe_truncate(Bad()) == {}


# ---------------------------------------------------------------------------
# LangGraphMonitorCallback (sync)
# ---------------------------------------------------------------------------


GRAPH_ID = "test_graph"
THREAD_ID = "thread_1"


class TestSyncCallback:
    def _make(self) -> tuple[LangGraphMonitorCallback, MagicMock]:
        db = mock_db()
        cb = LangGraphMonitorCallback(graph_id=GRAPH_ID, thread_id=THREAD_ID, db=db)
        return cb, db

    def test_on_chain_start_root_creates_run(self):
        cb, db = self._make()
        run_id = make_uuid()
        cb.on_chain_start({}, {}, run_id=run_id, parent_run_id=None)
        db.create_run.assert_called_once_with(str(run_id), GRAPH_ID, THREAD_ID)
        assert cb._run_id == str(run_id)

    def test_on_chain_start_node_inserts_event(self):
        cb, db = self._make()
        root_id = make_uuid()
        node_id = make_uuid()
        cb.on_chain_start({}, {}, run_id=root_id, parent_run_id=None)
        cb.on_chain_start({"name": "my_node"}, {"x": 1}, run_id=node_id, parent_run_id=root_id)
        db.insert_event.assert_called_once()
        kwargs = db.insert_event.call_args.kwargs
        assert kwargs["event_type"] == "node_start"
        assert kwargs["node_name"] == "my_node"

    def test_on_chain_end_root_completes_run(self):
        cb, db = self._make()
        run_id = make_uuid()
        cb.on_chain_start({}, {}, run_id=run_id, parent_run_id=None)
        cb.on_chain_end({}, run_id=run_id, parent_run_id=None)
        db.complete_run.assert_called_once_with(str(run_id))

    def test_on_chain_end_node_inserts_event_with_latency(self):
        cb, db = self._make()
        root_id = make_uuid()
        node_id = make_uuid()
        cb.on_chain_start({}, {}, run_id=root_id, parent_run_id=None)
        cb.on_chain_start({"name": "n"}, {}, run_id=node_id, parent_run_id=root_id)
        cb.on_chain_end({}, run_id=node_id, parent_run_id=root_id, name="n")
        kwargs = db.insert_event.call_args_list[-1].kwargs
        assert kwargs["event_type"] == "node_end"
        assert kwargs["latency_ms"] is not None

    def test_on_chain_error_root_fails_run(self):
        cb, db = self._make()
        run_id = make_uuid()
        cb.on_chain_start({}, {}, run_id=run_id, parent_run_id=None)
        cb.on_chain_error(ValueError("bad"), run_id=run_id, parent_run_id=None)
        db.fail_run.assert_called_once()
        assert "ValueError" in db.fail_run.call_args.args[1]

    def test_on_chain_error_node_inserts_error_event(self):
        cb, db = self._make()
        root_id = make_uuid()
        node_id = make_uuid()
        cb.on_chain_start({}, {}, run_id=root_id, parent_run_id=None)
        cb.on_chain_start({"name": "n"}, {}, run_id=node_id, parent_run_id=root_id)
        cb.on_chain_error(RuntimeError("oops"), run_id=node_id, parent_run_id=root_id, name="n")
        kwargs = db.insert_event.call_args_list[-1].kwargs
        assert kwargs["event_type"] == "error"
        assert "RuntimeError" in kwargs["error"]

    def test_on_tool_start_inserts_tool_call_event(self):
        cb, db = self._make()
        root_id = make_uuid()
        tool_id = make_uuid()
        cb.on_chain_start({}, {}, run_id=root_id, parent_run_id=None)
        cb.on_tool_start({"name": "search"}, "query", run_id=tool_id)
        kwargs = db.insert_event.call_args.kwargs
        assert kwargs["event_type"] == "tool_call"
        assert kwargs["node_name"] == "search"

    def test_on_tool_end_inserts_tool_result_event(self):
        cb, db = self._make()
        root_id = make_uuid()
        tool_id = make_uuid()
        cb.on_chain_start({}, {}, run_id=root_id, parent_run_id=None)
        cb.on_tool_start({"name": "search"}, "query", run_id=tool_id)
        cb.on_tool_end("result", run_id=tool_id, name="search")
        kwargs = db.insert_event.call_args.kwargs
        assert kwargs["event_type"] == "tool_result"

    def test_on_tool_error_inserts_error_event(self):
        cb, db = self._make()
        root_id = make_uuid()
        tool_id = make_uuid()
        cb.on_chain_start({}, {}, run_id=root_id, parent_run_id=None)
        cb.on_tool_start({"name": "search"}, "query", run_id=tool_id)
        cb.on_tool_error(OSError("network"), run_id=tool_id, name="search")
        kwargs = db.insert_event.call_args.kwargs
        assert kwargs["event_type"] == "error"
        assert "OSError" in kwargs["error"]


# ---------------------------------------------------------------------------
# AsyncLangGraphMonitorCallback
# ---------------------------------------------------------------------------


class TestAsyncCallback:
    def _make(self) -> tuple[AsyncLangGraphMonitorCallback, MagicMock]:
        db = mock_db()
        cb = AsyncLangGraphMonitorCallback(graph_id=GRAPH_ID, thread_id=THREAD_ID, db=db)
        return cb, db

    async def test_on_chain_start_root_creates_run(self):
        cb, db = self._make()
        run_id = make_uuid()
        await cb.on_chain_start({}, {}, run_id=run_id, parent_run_id=None)
        db.create_run.assert_called_once_with(str(run_id), GRAPH_ID, THREAD_ID)

    async def test_on_chain_end_root_completes_run(self):
        cb, db = self._make()
        run_id = make_uuid()
        await cb.on_chain_start({}, {}, run_id=run_id, parent_run_id=None)
        await cb.on_chain_end({}, run_id=run_id, parent_run_id=None)
        db.complete_run.assert_called_once_with(str(run_id))

    async def test_on_chain_error_root_fails_run(self):
        cb, db = self._make()
        run_id = make_uuid()
        await cb.on_chain_start({}, {}, run_id=run_id, parent_run_id=None)
        await cb.on_chain_error(ValueError("bad"), run_id=run_id, parent_run_id=None)
        db.fail_run.assert_called_once()

    async def test_on_tool_start_and_end(self):
        cb, db = self._make()
        root_id = make_uuid()
        tool_id = make_uuid()
        await cb.on_chain_start({}, {}, run_id=root_id, parent_run_id=None)
        await cb.on_tool_start({"name": "calc"}, "1+1", run_id=tool_id)
        await cb.on_tool_end("2", run_id=tool_id, name="calc")
        events = [c.kwargs["event_type"] for c in db.insert_event.call_args_list]
        assert "tool_call" in events
        assert "tool_result" in events
