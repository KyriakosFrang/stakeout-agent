from __future__ import annotations

from abc import ABC, abstractmethod


class AbstractMonitorDB(ABC):
    @abstractmethod
    def create_run(self, run_id: str, graph_id: str, thread_id: str) -> None: ...

    @abstractmethod
    def complete_run(self, run_id: str) -> None: ...

    @abstractmethod
    def fail_run(self, run_id: str, error: str) -> None: ...

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
    ) -> None: ...
