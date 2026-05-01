from __future__ import annotations

import asyncio
from typing import Any
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler, BaseCallbackHandler

from stakeout_agent.db import MonitorDB

from .base import _MonitorBase


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
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, lambda: self._handle_chain_start(serialized, inputs, run_id, parent_run_id, **kwargs)
        )

    async def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: self._handle_chain_end(outputs, run_id, parent_run_id, **kwargs))

    async def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: self._handle_chain_error(error, run_id, parent_run_id, **kwargs))

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,  # noqa: ARG002
        **kwargs: Any,
    ) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: self._handle_tool_start(serialized, input_str, run_id, **kwargs))

    async def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: self._handle_tool_end(output, run_id, **kwargs))

    async def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: self._handle_tool_error(error, run_id, **kwargs))
