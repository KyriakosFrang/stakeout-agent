from __future__ import annotations

import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler, BaseCallbackHandler

from stakeout_agent.db import MonitorDB


class _MonitorBase:
    """Shared state and logic for both sync and async monitor callbacks."""

    def __init__(self, graph_id: str, thread_id: str, db: MonitorDB | None = None):
        self.graph_id = graph_id
        self.thread_id = thread_id
        self.db = db or MonitorDB()

        self._run_id: str | None = None
        self._node_start_times: dict[str, float] = {}
        self._tool_start_times: dict[str, float] = {}

    def _handle_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: dict[str, Any],
        run_id: UUID,
        parent_run_id: UUID | None,
        **kwargs: Any,
    ) -> None:
        run_id_str = str(run_id)
        if parent_run_id is None:
            self._run_id = run_id_str
            self.db.create_run(run_id_str, self.graph_id, self.thread_id)
        else:
            node_name = self._extract_name(serialized, kwargs)
            self._node_start_times[run_id_str] = time.monotonic()
            self.db.insert_event(
                run_id=self._run_id,
                graph_id=self.graph_id,
                event_type="node_start",
                node_name=node_name,
                payload={"inputs": self._safe_truncate(inputs)},
            )

    def _handle_chain_end(
        self,
        outputs: dict[str, Any],
        run_id: UUID,
        parent_run_id: UUID | None,
        **kwargs: Any,
    ) -> None:
        run_id_str = str(run_id)
        if parent_run_id is None:
            self.db.complete_run(self._run_id)
        else:
            latency = self._pop_latency(self._node_start_times, run_id_str)
            node_name = kwargs.get("name", "unknown")
            self.db.insert_event(
                run_id=self._run_id,
                graph_id=self.graph_id,
                event_type="node_end",
                node_name=node_name,
                latency_ms=latency,
                payload={"outputs": self._safe_truncate(outputs)},
            )

    def _handle_chain_error(
        self,
        error: BaseException,
        run_id: UUID,
        parent_run_id: UUID | None,
        **kwargs: Any,
    ) -> None:
        run_id_str = str(run_id)
        error_str = f"{type(error).__name__}: {str(error)}"
        if parent_run_id is None:
            self.db.fail_run(self._run_id, error_str)
        else:
            latency = self._pop_latency(self._node_start_times, run_id_str)
            node_name = kwargs.get("name", "unknown")
            self.db.insert_event(
                run_id=self._run_id,
                graph_id=self.graph_id,
                event_type="error",
                node_name=node_name,
                latency_ms=latency,
                error=error_str,
            )

    def _handle_tool_start(
        self,
        serialized: dict[str, Any] | None,
        input_str: str,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        run_id_str = str(run_id)
        tool_name = serialized.get("name", "unknown_tool") if serialized else kwargs.get("name", "unknown_tool")
        self._tool_start_times[run_id_str] = time.monotonic()
        self.db.insert_event(
            run_id=self._run_id,
            graph_id=self.graph_id,
            event_type="tool_call",
            node_name=tool_name,
            payload={"input": input_str[:500]},
        )

    def _handle_tool_end(self, output: Any, run_id: UUID, **kwargs: Any) -> None:
        run_id_str = str(run_id)
        latency = self._pop_latency(self._tool_start_times, run_id_str)
        tool_name = kwargs.get("name", "unknown_tool")
        self.db.insert_event(
            run_id=self._run_id,
            graph_id=self.graph_id,
            event_type="tool_result",
            node_name=tool_name,
            latency_ms=latency,
            payload={"output": str(output)[:500]},
        )

    def _handle_tool_error(self, error: BaseException, run_id: UUID, **kwargs: Any) -> None:
        run_id_str = str(run_id)
        latency = self._pop_latency(self._tool_start_times, run_id_str)
        tool_name = kwargs.get("name", "unknown_tool")
        self.db.insert_event(
            run_id=self._run_id,
            graph_id=self.graph_id,
            event_type="error",
            node_name=tool_name,
            latency_ms=latency,
            error=f"{type(error).__name__}: {str(error)}",
        )

    @staticmethod
    def _extract_name(serialized: dict | None, kwargs: dict) -> str:
        if serialized:
            return serialized.get("name") or serialized.get("id", ["unknown"])[-1] or kwargs.get("name", "unknown")
        return kwargs.get("name", "unknown")

    @staticmethod
    def _pop_latency(store: dict[str, float], key: str) -> float | None:
        start = store.pop(key, None)
        if start is None:
            return None
        return round((time.monotonic() - start) * 1000, 2)

    @staticmethod
    def _safe_truncate(data: Any, max_len: int = 500) -> Any:
        try:
            text = str(data)
            return text[:max_len] if len(text) > max_len else data
        except Exception:
            return {}


class LangGraphMonitorCallback(_MonitorBase, BaseCallbackHandler):
    """Sync monitor for use with graph.invoke().

    Usage:
        monitor = LangGraphMonitorCallback(graph_id="my_graph", thread_id="thread_123")
        graph.invoke(inputs, config={"callbacks": [monitor]})
    """

    def __init__(self, graph_id: str, thread_id: str, db: MonitorDB | None = None):
        _MonitorBase.__init__(self, graph_id, thread_id, db)
        BaseCallbackHandler.__init__(self)

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,  # noqa: ARG002
        **kwargs: Any,
    ) -> None:
        self._handle_chain_start(serialized, inputs, run_id, parent_run_id, **kwargs)

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._handle_chain_end(outputs, run_id, parent_run_id, **kwargs)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._handle_chain_error(error, run_id, parent_run_id, **kwargs)

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,  # noqa: ARG002
        **kwargs: Any,
    ) -> None:
        self._handle_tool_start(serialized, input_str, run_id, **kwargs)

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._handle_tool_end(output, run_id, **kwargs)

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._handle_tool_error(error, run_id, **kwargs)


class AsyncLangGraphMonitorCallback(_MonitorBase, AsyncCallbackHandler):
    """Async monitor for use with graph.ainvoke() / graph.astream().

    Usage:
        monitor = AsyncLangGraphMonitorCallback(graph_id="my_graph", thread_id="thread_123")
        await graph.ainvoke(inputs, config={"callbacks": [monitor]})
    """

    def __init__(self, graph_id: str, thread_id: str, db: MonitorDB | None = None):
        _MonitorBase.__init__(self, graph_id, thread_id, db)
        AsyncCallbackHandler.__init__(self)

    async def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,  # noqa: ARG002
        **kwargs: Any,
    ) -> None:
        self._handle_chain_start(serialized, inputs, run_id, parent_run_id, **kwargs)

    async def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._handle_chain_end(outputs, run_id, parent_run_id, **kwargs)

    async def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._handle_chain_error(error, run_id, parent_run_id, **kwargs)

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,  # noqa: ARG002
        **kwargs: Any,
    ) -> None:
        self._handle_tool_start(serialized, input_str, run_id, **kwargs)

    async def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._handle_tool_end(output, run_id, **kwargs)

    async def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._handle_tool_error(error, run_id, **kwargs)
