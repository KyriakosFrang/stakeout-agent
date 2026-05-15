"""Tests for stakeout_agent.writer.BufferedWriter."""

from __future__ import annotations

import json
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from stakeout_agent.writer import BufferedWriter


def _mock_backend() -> MagicMock:
    backend = MagicMock()
    backend.create_run.return_value = None
    backend.complete_run.return_value = None
    backend.fail_run.return_value = None
    backend.insert_event.return_value = None
    return backend


# ---------------------------------------------------------------------------
# Non-blocking enqueue
# ---------------------------------------------------------------------------


class TestNonBlocking:
    def test_create_run_returns_before_backend_executes(self):
        """Enqueue must return without waiting for the backend call."""
        barrier = {"reached": False}

        def slow_create(*a, **kw):
            time.sleep(0.1)
            barrier["reached"] = True

        backend = _mock_backend()
        backend.create_run.side_effect = slow_create

        with BufferedWriter(backend=backend) as writer:
            start = time.monotonic()
            writer.create_run("r1", "g", "t")
            elapsed = time.monotonic() - start
            # The enqueue itself must be fast — well under the 100 ms sleep
            assert elapsed < 0.05

        assert barrier["reached"]  # worker did execute it eventually

    def test_all_methods_are_non_blocking(self):
        """All four AbstractMonitorDB methods enqueue without sleeping."""
        backend = _mock_backend()

        def slow(*a, **kw):
            time.sleep(0.05)

        backend.create_run.side_effect = slow
        backend.complete_run.side_effect = slow
        backend.fail_run.side_effect = slow
        backend.insert_event.side_effect = slow

        with BufferedWriter(backend=backend) as writer:
            start = time.monotonic()
            writer.create_run("r", "g", "t")
            writer.complete_run("r")
            writer.fail_run("r", "boom")
            writer.insert_event("r", "g", "node_start", "n")
            elapsed = time.monotonic() - start
            assert elapsed < 0.05


# ---------------------------------------------------------------------------
# Correct arguments forwarded to backend
# ---------------------------------------------------------------------------


class TestArgumentForwarding:
    def test_create_run_forwards_all_args(self):
        backend = _mock_backend()
        with BufferedWriter(backend=backend) as writer:
            writer.create_run("run-1", "my_graph", "thread-42", run_inputs="hi", prompt_version="v2")

        backend.create_run.assert_called_once_with(
            "run-1", "my_graph", "thread-42", run_inputs="hi", parent_run_id=None, prompt_version="v2"
        )

    def test_complete_run_forwards_token_args(self):
        backend = _mock_backend()
        with BufferedWriter(backend=backend) as writer:
            writer.complete_run("r", total_input_tokens=10, total_output_tokens=5, estimated_cost_usd=0.001)

        backend.complete_run.assert_called_once_with(
            "r",
            total_input_tokens=10,
            total_output_tokens=5,
            estimated_cost_usd=0.001,
            total_cache_read_tokens=None,
            total_cache_creation_tokens=None,
        )

    def test_fail_run_forwards_error(self):
        backend = _mock_backend()
        with BufferedWriter(backend=backend) as writer:
            writer.fail_run("r", "something broke")

        backend.fail_run.assert_called_once_with("r", "something broke")

    def test_insert_event_forwards_all_optional_args(self):
        backend = _mock_backend()
        with BufferedWriter(backend=backend) as writer:
            writer.insert_event(
                "r",
                "g",
                "node_end",
                "my_node",
                latency_ms=12.5,
                payload={"k": "v"},
                input_tokens=7,
                model="claude-3",
            )

        backend.insert_event.assert_called_once_with(
            "r",
            "g",
            "node_end",
            "my_node",
            latency_ms=12.5,
            payload={"k": "v"},
            error=None,
            messages=None,
            input_tokens=7,
            output_tokens=None,
            model="claude-3",
            llm_input=None,
            llm_output=None,
            cache_read_tokens=None,
            cache_creation_tokens=None,
        )


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------


class TestRetryLogic:
    def test_retries_on_failure_then_succeeds(self):
        """Backend raises on first 2 attempts, succeeds on 3rd."""
        backend = _mock_backend()
        call_count = 0

        def flaky(*a, **kw):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise OSError("transient")

        backend.create_run.side_effect = flaky

        with patch("stakeout_agent.writer.time.sleep"):
            with BufferedWriter(backend=backend, max_retries=3) as writer:
                writer.create_run("r", "g", "t")

        assert call_count == 3
        assert writer.dropped_events == 0

    def test_exhausted_retries_increment_dropped_events(self):
        backend = _mock_backend()
        backend.create_run.side_effect = OSError("always fails")

        with patch("stakeout_agent.writer.time.sleep"):
            with BufferedWriter(backend=backend, max_retries=3, dlq_path="/dev/null") as writer:
                writer.create_run("r", "g", "t")

        assert writer.dropped_events == 1
        assert backend.create_run.call_count == 3

    def test_backoff_delays_are_applied(self):
        backend = _mock_backend()
        backend.fail_run.side_effect = OSError("boom")
        sleep_calls: list[float] = []

        def capture_sleep(secs):
            sleep_calls.append(secs)

        with patch("stakeout_agent.writer.time.sleep", side_effect=capture_sleep):
            with BufferedWriter(backend=backend, max_retries=3, dlq_path="/dev/null") as writer:
                writer.fail_run("r", "err")

        # Expect two sleeps (between attempt 1→2 and 2→3); no sleep after final failure
        assert len(sleep_calls) == 2
        assert sleep_calls[0] == pytest.approx(0.5)
        assert sleep_calls[1] == pytest.approx(1.0)

    def test_no_retry_when_max_retries_is_one(self):
        backend = _mock_backend()
        backend.complete_run.side_effect = RuntimeError("instant fail")

        with patch("stakeout_agent.writer.time.sleep"):
            with BufferedWriter(backend=backend, max_retries=1, dlq_path="/dev/null") as writer:
                writer.complete_run("r")

        assert backend.complete_run.call_count == 1
        assert writer.dropped_events == 1


# ---------------------------------------------------------------------------
# Dead-letter queue
# ---------------------------------------------------------------------------


class TestDeadLetterQueue:
    def test_dlq_written_on_exhausted_retries(self, tmp_path):
        dlq = tmp_path / "dlq.jsonl"
        backend = _mock_backend()
        backend.insert_event.side_effect = RuntimeError("perm fail")

        with patch("stakeout_agent.writer.time.sleep"):
            with BufferedWriter(backend=backend, max_retries=2, dlq_path=str(dlq)) as writer:
                writer.insert_event("r1", "g", "node_start", "n")

        assert dlq.exists()
        lines = dlq.read_text().strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["method"] == "insert_event"
        assert "RuntimeError" in entry["error"]
        assert "timestamp" in entry

    def test_dlq_accumulates_multiple_failures(self, tmp_path):
        dlq = tmp_path / "dlq.jsonl"
        backend = _mock_backend()
        backend.create_run.side_effect = RuntimeError("fail")
        backend.fail_run.side_effect = RuntimeError("fail")

        with patch("stakeout_agent.writer.time.sleep"):
            with BufferedWriter(backend=backend, max_retries=1, dlq_path=str(dlq)) as writer:
                writer.create_run("r1", "g", "t")
                writer.fail_run("r2", "oops")

        lines = dlq.read_text().strip().splitlines()
        assert len(lines) == 2
        methods = {json.loads(line)["method"] for line in lines}
        assert methods == {"create_run", "fail_run"}

    def test_dlq_entry_includes_original_payload(self, tmp_path):
        dlq = tmp_path / "dlq.jsonl"
        backend = _mock_backend()
        backend.create_run.side_effect = OSError("gone")

        with patch("stakeout_agent.writer.time.sleep"):
            with BufferedWriter(backend=backend, max_retries=1, dlq_path=str(dlq)) as writer:
                writer.create_run("run-xyz", "graph-1", "thread-99")

        entry = json.loads(dlq.read_text().strip())
        assert entry["args"][0] == "run-xyz"
        assert entry["args"][1] == "graph-1"

    def test_dlq_path_from_env_var(self, tmp_path, monkeypatch):
        dlq = tmp_path / "env_dlq.jsonl"
        monkeypatch.setenv("STAKEOUT_DLQ_PATH", str(dlq))
        backend = _mock_backend()
        backend.create_run.side_effect = RuntimeError("fail")

        with patch("stakeout_agent.writer.time.sleep"):
            with BufferedWriter(backend=backend, max_retries=1) as writer:
                writer.create_run("r", "g", "t")

        assert dlq.exists()


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------


class TestGracefulShutdown:
    def test_close_flushes_all_pending_writes(self):
        backend = _mock_backend()
        processed: list[str] = []

        def record_create(run_id, *a, **kw):
            processed.append(run_id)

        backend.create_run.side_effect = record_create

        writer = BufferedWriter(backend=backend)
        for i in range(20):
            writer.create_run(f"r{i}", "g", "t")
        writer.close()

        assert len(processed) == 20

    def test_context_manager_calls_close(self):
        backend = _mock_backend()
        with BufferedWriter(backend=backend) as writer:
            writer.create_run("r", "g", "t")

        # After __exit__, the worker has stopped — calling close() again should be safe
        # and the backend must have been called exactly once
        backend.create_run.assert_called_once()

    def test_close_with_timeout_does_not_raise(self):
        backend = _mock_backend()
        writer = BufferedWriter(backend=backend)
        writer.close(timeout=5.0)  # must not raise even with generous timeout


# ---------------------------------------------------------------------------
# Queue-full behaviour
# ---------------------------------------------------------------------------


class TestQueueFull:
    def test_queue_full_drops_and_increments_counter(self):
        backend = _mock_backend()
        # Block the worker so the queue fills up
        gate = threading.Event()

        def blocking_create(*a, **kw):
            gate.wait()

        backend.create_run.side_effect = blocking_create

        writer = BufferedWriter(backend=backend, max_queue_size=2)
        # First item goes in and gets picked up by worker (blocks)
        writer.create_run("r0", "g", "t")
        time.sleep(0.05)  # give worker time to pick r0 off the queue
        # Fill the queue with max_queue_size items
        writer.create_run("r1", "g", "t")
        writer.create_run("r2", "g", "t")
        # This one should be dropped
        writer.create_run("r3", "g", "t")

        dropped = writer.dropped_events
        gate.set()  # unblock worker
        writer.close()

        assert dropped >= 1


# ---------------------------------------------------------------------------
# dropped_events counter — general
# ---------------------------------------------------------------------------


class TestDroppedEventsCounter:
    def test_counter_zero_on_all_success(self):
        backend = _mock_backend()
        with BufferedWriter(backend=backend) as writer:
            writer.create_run("r", "g", "t")
            writer.complete_run("r")
            writer.fail_run("r2", "err")
            writer.insert_event("r", "g", "e", "n")

        assert writer.dropped_events == 0

    def test_counter_accumulates_across_multiple_failures(self, tmp_path):
        dlq = tmp_path / "dlq.jsonl"
        backend = _mock_backend()
        backend.create_run.side_effect = RuntimeError("fail")
        backend.fail_run.side_effect = RuntimeError("fail")

        with patch("stakeout_agent.writer.time.sleep"):
            with BufferedWriter(backend=backend, max_retries=1, dlq_path=str(dlq)) as writer:
                writer.create_run("r1", "g", "t")
                writer.fail_run("r2", "oops")

        assert writer.dropped_events == 2


# ---------------------------------------------------------------------------
# Works with both sync and async callback variants (smoke test)
# ---------------------------------------------------------------------------


class TestCallbackIntegration:
    """Verify BufferedWriter is accepted as a db= argument to callback handlers."""

    def test_accepts_buffered_writer_as_db(self):
        """LangGraphMonitorCallback should store the BufferedWriter as its db."""
        pytest.importorskip("langchain_core")
        from stakeout_agent.callback_handler.langgraph import LangGraphMonitorCallback

        backend = _mock_backend()
        writer = BufferedWriter(backend=backend)
        cb = LangGraphMonitorCallback(graph_id="g", thread_id="t", db=writer)
        assert cb.db is writer
        writer.close()

    def test_async_variant_accepts_buffered_writer(self):
        pytest.importorskip("langchain_core")
        from stakeout_agent.callback_handler.langgraph import AsyncLangGraphMonitorCallback

        backend = _mock_backend()
        writer = BufferedWriter(backend=backend)
        cb = AsyncLangGraphMonitorCallback(graph_id="g", thread_id="t", db=writer)
        assert cb.db is writer
        writer.close()
