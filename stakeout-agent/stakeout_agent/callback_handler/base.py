from __future__ import annotations

import logging
import time
from typing import Any
from uuid import UUID

from stakeout_agent.backends.base import AbstractMonitorDB

_logger = logging.getLogger(__name__)


class _MonitorBase:
    """Shared state and logic reused by all framework-specific callback handlers."""

    def __init__(self, graph_id: str, thread_id: str, db: AbstractMonitorDB | None = None):
        self.graph_id = graph_id
        self.thread_id = thread_id
        if db is None:
            from stakeout_agent.backends import get_backend

            db = get_backend()
        self.db = db
        self._log = logging.LoggerAdapter(_logger, {"graph_id": graph_id, "thread_id": thread_id})

        self._run_id: str | None = None
        self._node_start_times: dict[str, float] = {}
        self._node_names: dict[str, str] = {}
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
            self._log.debug("run started run_id=%s", run_id_str)
            self.db.create_run(run_id_str, self.graph_id, self.thread_id)
        else:
            node_name = self._extract_name(serialized, kwargs)
            self._node_start_times[run_id_str] = time.monotonic()
            self._node_names[run_id_str] = node_name
            self._log.debug("node_start node=%s run_id=%s", node_name, self._run_id)
            self.db.insert_event(
                run_id=self._run_id,
                graph_id=self.graph_id,
                event_type="node_start",
                node_name=node_name,
                payload={"inputs": self._safe_truncate(inputs)},
                messages=self._extract_messages(inputs),
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
            self._log.debug("run completed run_id=%s", self._run_id)
            self.db.complete_run(self._run_id)
        else:
            latency = self._pop_latency(self._node_start_times, run_id_str)
            node_name = self._node_names.pop(run_id_str, "unknown")
            self._log.debug("node_end node=%s latency_ms=%s run_id=%s", node_name, latency, self._run_id)
            self.db.insert_event(
                run_id=self._run_id,
                graph_id=self.graph_id,
                event_type="node_end",
                node_name=node_name,
                latency_ms=latency,
                payload={"outputs": self._safe_truncate(outputs)},
                messages=self._extract_messages(outputs),
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
            self._log.warning("run failed run_id=%s error=%s", self._run_id, error_str)
            self.db.fail_run(self._run_id, error_str)
        else:
            latency = self._pop_latency(self._node_start_times, run_id_str)
            node_name = self._node_names.pop(run_id_str, "unknown")
            self._log.warning("node error node=%s run_id=%s error=%s", node_name, self._run_id, error_str)
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
        self._log.debug("tool_call tool=%s run_id=%s", tool_name, self._run_id)
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
        self._log.debug("tool_result tool=%s latency_ms=%s run_id=%s", tool_name, latency, self._run_id)
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
        self._log.warning("tool error tool=%s run_id=%s error=%s", tool_name, self._run_id, f"{type(error).__name__}: {error}")
        self.db.insert_event(
            run_id=self._run_id,
            graph_id=self.graph_id,
            event_type="error",
            node_name=tool_name,
            latency_ms=latency,
            error=f"{type(error).__name__}: {str(error)}",
        )

    @staticmethod
    def _extract_messages(data: Any) -> list[dict] | None:
        """Extract a messages list from a LangGraph state dict into plain {role, content} dicts.

        Returns None when the state has no messages field, so callers can omit the field entirely.
        Handles both LangChain BaseMessage objects and already-serialised dicts.
        """
        if not isinstance(data, dict):
            return None
        msgs = data.get("messages")
        if not isinstance(msgs, list) or not msgs:
            return None
        _ROLE_MAP = {"human": "human", "ai": "assistant", "system": "system", "tool": "tool"}
        result = []
        for m in msgs:
            if hasattr(m, "type") and hasattr(m, "content"):
                # LangChain BaseMessage subclass
                role = _ROLE_MAP.get(m.type, m.type)
                content = m.content if isinstance(m.content, str) else str(m.content)
                result.append({"role": role, "content": content[:500]})
            elif isinstance(m, dict) and "role" in m:
                result.append({"role": m["role"], "content": str(m.get("content", ""))[:500]})
        return result or None

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
