from __future__ import annotations

from abc import ABC, abstractmethod


class AbstractQueryDB(ABC):
    """Read-only query interface implemented by persistent backends (MongoDB, Postgres).

    The OTEL backend has no storage of its own and does not implement this.
    """

    @abstractmethod
    def query_recent_runs(self, graph_id: str | None, limit: int) -> list[dict]:
        """Return the *limit* most-recent runs, optionally filtered by graph_id."""
        ...

    @abstractmethod
    def query_run_detail(self, run_id: str) -> dict | None:
        """Return a run document plus all its events, or None if not found."""
        ...

    @abstractmethod
    def query_failed_runs(self, graph_id: str | None, since_ts: float) -> list[dict]:
        """Return failed runs with started_at >= since_ts (Unix timestamp)."""
        ...

    @abstractmethod
    def query_slow_runs(self, graph_id: str | None, threshold_ms: float, since_ts: float) -> list[dict]:
        """Return completed runs whose wall-clock duration exceeds threshold_ms."""
        ...

    @abstractmethod
    def query_run_stats(self, graph_id: str | None, since_ts: float) -> dict:
        """Return aggregate stats (error_rate, p50/p95 latency, cost) since since_ts."""
        ...

    @abstractmethod
    def query_runs_by_output(self, graph_id: str | None, text: str, limit: int) -> list[dict]:
        """Return runs whose run_inputs field contains *text* (case-insensitive)."""
        ...


class AbstractMonitorDB(ABC):
    @abstractmethod
    def create_run(
        self,
        run_id: str,
        graph_id: str,
        thread_id: str,
        run_inputs: str | None = None,
        parent_run_id: str | None = None,
        prompt_version: str | None = None,
        environment: str | None = None,
    ) -> None: ...

    @abstractmethod
    def complete_run(
        self,
        run_id: str,
        total_input_tokens: int | None = None,
        total_output_tokens: int | None = None,
        estimated_cost_usd: float | None = None,
        total_cache_read_tokens: int | None = None,
        total_cache_creation_tokens: int | None = None,
    ) -> None: ...

    @abstractmethod
    def fail_run(self, run_id: str, error: str) -> None: ...

    @abstractmethod
    def prune_runs(self, older_than_days: int) -> int:
        """Delete runs (and their events) whose started_at is older than *older_than_days* days.

        Returns the number of runs deleted.
        """
        ...

    @abstractmethod
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
    ) -> None: ...
