"""BufferedWriter — non-blocking wrapper around any AbstractMonitorDB.

All backend calls are enqueued and executed by a background thread, so the
LLM hot path never blocks on I/O. Failed writes are retried with exponential
backoff; writes that exhaust all retries are appended to a dead-letter queue
(DLQ) file for later replay or alerting.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import queue
import threading
import time
from datetime import datetime, timezone
from typing import Any

from stakeout_agent.backends.base import AbstractMonitorDB

_log = logging.getLogger(__name__)

# Sentinel placed on the queue to signal the worker to stop.
_STOP = object()


@dataclasses.dataclass
class _WriteTask:
    method: str
    args: tuple
    kwargs: dict


class BufferedWriter(AbstractMonitorDB):
    """Non-blocking wrapper around any AbstractMonitorDB.

    Enqueues all writes and executes them on a background daemon thread.
    Failed writes are retried with exponential backoff up to *max_retries*
    times; writes that still fail are appended to the DLQ file as
    newline-delimited JSON and counted in ``dropped_events``.

    Usage::

        with BufferedWriter(backend=MongoMonitorDB()) as writer:
            cb = LangGraphMonitorCallback(graph_id="g", thread_id="t", db=writer)
            # … run your graph …
        # close() flushes remaining writes before returning

    The worker thread is a daemon so the process can exit if ``close()`` is
    not called; any still-queued writes will then be lost. Call ``close()``
    (or use the context manager) when guaranteed delivery matters.
    """

    def __init__(
        self,
        backend: AbstractMonitorDB,
        max_queue_size: int = 10_000,
        max_retries: int = 3,
        dlq_path: str | None = None,
    ) -> None:
        self._backend = backend
        self._max_retries = max_retries
        self._dlq_path = dlq_path or os.getenv("STAKEOUT_DLQ_PATH", "stakeout_dlq.jsonl")
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=max_queue_size)
        self._dropped = 0
        self._lock = threading.Lock()
        self._worker_thread = threading.Thread(
            target=self._run_worker,
            daemon=True,
            name="stakeout-buffered-writer",
        )
        self._worker_thread.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def dropped_events(self) -> int:
        """Total writes that were ultimately not persisted (DLQ + queue-full drops)."""
        with self._lock:
            return self._dropped

    def close(self, timeout: float | None = None) -> None:
        """Flush all pending writes and stop the background worker.

        Blocks until the worker finishes or *timeout* seconds elapse.
        """
        self._queue.put(_STOP)
        self._worker_thread.join(timeout=timeout)

    def __enter__(self) -> BufferedWriter:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # AbstractMonitorDB — all return immediately after enqueue
    # ------------------------------------------------------------------

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
        self._enqueue(
            "create_run",
            run_id,
            graph_id,
            thread_id,
            run_inputs=run_inputs,
            parent_run_id=parent_run_id,
            prompt_version=prompt_version,
            environment=environment,
        )

    def complete_run(
        self,
        run_id: str,
        total_input_tokens: int | None = None,
        total_output_tokens: int | None = None,
        estimated_cost_usd: float | None = None,
        total_cache_read_tokens: int | None = None,
        total_cache_creation_tokens: int | None = None,
    ) -> None:
        self._enqueue(
            "complete_run",
            run_id,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            estimated_cost_usd=estimated_cost_usd,
            total_cache_read_tokens=total_cache_read_tokens,
            total_cache_creation_tokens=total_cache_creation_tokens,
        )

    def fail_run(self, run_id: str, error: str) -> None:
        self._enqueue("fail_run", run_id, error)

    def prune_runs(self, older_than_days: int) -> int:
        return self._backend.prune_runs(older_than_days)

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
        self._enqueue(
            "insert_event",
            run_id,
            graph_id,
            event_type,
            node_name,
            latency_ms=latency_ms,
            payload=payload,
            error=error,
            messages=messages,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
            llm_input=llm_input,
            llm_output=llm_output,
            cache_read_tokens=cache_read_tokens,
            cache_creation_tokens=cache_creation_tokens,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _enqueue(self, method: str, *args: Any, **kwargs: Any) -> None:
        task = _WriteTask(method=method, args=args, kwargs=kwargs)
        try:
            self._queue.put_nowait(task)
        except queue.Full:
            _log.warning("BufferedWriter queue full — dropping %s", method)
            with self._lock:
                self._dropped += 1

    def _run_worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is _STOP:
                self._queue.task_done()
                break
            self._execute(item)
            self._queue.task_done()

    def _execute(self, task: _WriteTask) -> None:
        fn = getattr(self._backend, task.method)
        for attempt in range(1, self._max_retries + 1):
            try:
                fn(*task.args, **task.kwargs)
                return
            except Exception as exc:
                if attempt < self._max_retries:
                    delay = 0.5 * (2 ** (attempt - 1))
                    _log.warning(
                        "BufferedWriter: %s attempt %d/%d failed: %s — retrying in %.1fs",
                        task.method,
                        attempt,
                        self._max_retries,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
                else:
                    _log.error(
                        "BufferedWriter: %s failed after %d attempts: %s — writing to DLQ %s",
                        task.method,
                        self._max_retries,
                        exc,
                        self._dlq_path,
                    )
                    self._write_dlq(task, exc)
                    with self._lock:
                        self._dropped += 1

    def _write_dlq(self, task: _WriteTask, exc: Exception) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "method": task.method,
            "args": task.args,
            "kwargs": task.kwargs,
            "error": f"{type(exc).__name__}: {exc}",
        }
        try:
            with open(self._dlq_path, "a") as fh:
                fh.write(json.dumps(entry, default=str) + "\n")
        except Exception as dlq_exc:
            _log.error("BufferedWriter: failed to write to DLQ %s: %s", self._dlq_path, dlq_exc)
