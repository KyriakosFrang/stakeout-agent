from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any
from uuid import UUID

try:
    from langchain_core.callbacks import AsyncCallbackHandler, BaseCallbackHandler
    from langchain_core.outputs import LLMResult
except ImportError:

    class BaseCallbackHandler:  # type: ignore[no-redef]
        pass

    class AsyncCallbackHandler:  # type: ignore[no-redef]
        pass

    LLMResult = None  # type: ignore[assignment]

from stakeout_agent.backends.base import AbstractMonitorDB

from .base import _MonitorBase


class LangGraphMonitorCallback(_MonitorBase, BaseCallbackHandler):
    """Sync monitor for use with graph.invoke().

    A single instance can safely be shared across concurrent graph.invoke() calls —
    per-run state is keyed by the root run UUID, so concurrent invocations cannot
    interfere with each other.

    Usage:
        monitor = LangGraphMonitorCallback(graph_id="my_graph", thread_id="thread_123")
        graph.invoke(inputs, config={"callbacks": [monitor]})
    """

    def __init__(
        self,
        graph_id: str,
        thread_id: str,
        db: AbstractMonitorDB | None = None,
        pricing=None,
        token_extractor: Callable[[dict], tuple[int | None, int | None, str | None]] | None = None,
        capture_payloads: bool = True,
        max_payload_chars: int | None = None,
        parent_run_id: str | None = None,
        prompt_version: str | None = None,
    ):
        if LLMResult is None:
            raise ImportError(
                "langchain-core is required for LangGraphMonitorCallback. "
                "Install it with: pip install 'stakeout-agent[langgraph]'"
            )
        _MonitorBase.__init__(
            self,
            graph_id,
            thread_id,
            db,
            pricing=pricing,
            token_extractor=token_extractor,
            capture_payloads=capture_payloads,
            max_payload_chars=max_payload_chars,
            parent_run_id=parent_run_id,
            prompt_version=prompt_version,
        )
        BaseCallbackHandler.__init__(self)

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self._handle_chain_start(serialized, inputs, run_id, parent_run_id, tags=tags, **kwargs)

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
        parent_run_id: UUID | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._handle_tool_start(serialized, input_str, run_id, parent_run_id=parent_run_id, inputs=inputs, **kwargs)

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._handle_tool_end(output, run_id, **kwargs)

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._handle_tool_error(error, run_id, **kwargs)

    def on_retriever_start(
        self,
        serialized: dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._handle_retriever_start(serialized, query, run_id, parent_run_id=parent_run_id, **kwargs)

    def on_retriever_end(
        self,
        documents: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,  # noqa: ARG002
        **kwargs: Any,
    ) -> None:
        self._handle_retriever_end(documents, run_id, **kwargs)

    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,  # noqa: ARG002
        **kwargs: Any,
    ) -> None:
        self._handle_retriever_error(error, run_id, **kwargs)

    def on_llm_start(
        self,
        serialized: dict[str, Any],  # noqa: ARG002
        prompts: list[str],
        *,
        run_id: UUID,  # noqa: ARG002
        parent_run_id: UUID | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        formatted = [{"role": "user", "content": p} for p in prompts]
        self._handle_llm_start(formatted, parent_run_id)

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],  # noqa: ARG002
        messages: list[list[Any]],
        *,
        run_id: UUID,  # noqa: ARG002
        parent_run_id: UUID | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        _ROLE_MAP = {"human": "human", "ai": "assistant", "system": "system", "tool": "tool"}
        formatted = []
        for batch in messages:
            for m in batch:
                if hasattr(m, "type") and hasattr(m, "content"):
                    role = _ROLE_MAP.get(m.type, m.type)
                    content = m.content if isinstance(m.content, str) else str(m.content)
                    formatted.append({"role": role, "content": content})
                elif isinstance(m, dict) and "role" in m:
                    formatted.append({"role": m["role"], "content": str(m.get("content", ""))})
        self._handle_llm_start(formatted, parent_run_id)

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,  # noqa: ARG002
        parent_run_id: UUID | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        try:
            output_text = response.generations[0][0].text
        except (IndexError, AttributeError):
            output_text = None
        self._handle_llm_end(response.llm_output or {}, parent_run_id, output_text=output_text)


class AsyncLangGraphMonitorCallback(_MonitorBase, AsyncCallbackHandler):
    """Async monitor for use with graph.ainvoke() / graph.astream().

    A single instance can safely be shared across concurrent graph.ainvoke() calls —
    per-run state is keyed by the root run UUID, so concurrent invocations via
    asyncio.gather cannot interfere with each other.

    Usage:
        monitor = AsyncLangGraphMonitorCallback(graph_id="my_graph", thread_id="thread_123")
        await graph.ainvoke(inputs, config={"callbacks": [monitor]})
    """

    def __init__(
        self,
        graph_id: str,
        thread_id: str,
        db: AbstractMonitorDB | None = None,
        pricing=None,
        token_extractor: Callable[[dict], tuple[int | None, int | None, str | None]] | None = None,
        capture_payloads: bool = True,
        max_payload_chars: int | None = None,
        parent_run_id: str | None = None,
        prompt_version: str | None = None,
    ):
        if LLMResult is None:
            raise ImportError(
                "langchain-core is required for AsyncLangGraphMonitorCallback. "
                "Install it with: pip install 'stakeout-agent[langgraph]'"
            )
        _MonitorBase.__init__(
            self,
            graph_id,
            thread_id,
            db,
            pricing=pricing,
            token_extractor=token_extractor,
            capture_payloads=capture_payloads,
            max_payload_chars=max_payload_chars,
            parent_run_id=parent_run_id,
            prompt_version=prompt_version,
        )
        AsyncCallbackHandler.__init__(self)

    async def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, lambda: self._handle_chain_start(serialized, inputs, run_id, parent_run_id, tags=tags, **kwargs)
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
        parent_run_id: UUID | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: self._handle_tool_start(
                serialized, input_str, run_id, parent_run_id=parent_run_id, inputs=inputs, **kwargs
            ),
        )

    async def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: self._handle_tool_end(output, run_id, **kwargs))

    async def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: self._handle_tool_error(error, run_id, **kwargs))

    async def on_retriever_start(
        self,
        serialized: dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: self._handle_retriever_start(serialized, query, run_id, parent_run_id=parent_run_id, **kwargs),
        )

    async def on_retriever_end(
        self,
        documents: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,  # noqa: ARG002
        **kwargs: Any,
    ) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: self._handle_retriever_end(documents, run_id, **kwargs))

    async def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,  # noqa: ARG002
        **kwargs: Any,
    ) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: self._handle_retriever_error(error, run_id, **kwargs))

    async def on_llm_start(
        self,
        serialized: dict[str, Any],  # noqa: ARG002
        prompts: list[str],
        *,
        run_id: UUID,  # noqa: ARG002
        parent_run_id: UUID | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        formatted = [{"role": "user", "content": p} for p in prompts]
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: self._handle_llm_start(formatted, parent_run_id))

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],  # noqa: ARG002
        messages: list[list[Any]],
        *,
        run_id: UUID,  # noqa: ARG002
        parent_run_id: UUID | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        _ROLE_MAP = {"human": "human", "ai": "assistant", "system": "system", "tool": "tool"}
        formatted = []
        for batch in messages:
            for m in batch:
                if hasattr(m, "type") and hasattr(m, "content"):
                    role = _ROLE_MAP.get(m.type, m.type)
                    content = m.content if isinstance(m.content, str) else str(m.content)
                    formatted.append({"role": role, "content": content})
                elif isinstance(m, dict) and "role" in m:
                    formatted.append({"role": m["role"], "content": str(m.get("content", ""))})
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: self._handle_llm_start(formatted, parent_run_id))

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,  # noqa: ARG002
        parent_run_id: UUID | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        try:
            output_text = response.generations[0][0].text
        except (IndexError, AttributeError):
            output_text = None
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, lambda: self._handle_llm_end(response.llm_output or {}, parent_run_id, output_text=output_text)
        )
