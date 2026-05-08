from __future__ import annotations

import asyncio
import time
from typing import Any
from uuid import uuid4

try:
    from crewai.events.base_event_listener import BaseEventListener
    from crewai.events.types.crew_events import (
        CrewKickoffCompletedEvent,
        CrewKickoffFailedEvent,
        CrewKickoffStartedEvent,
    )
    from crewai.events.types.llm_events import LLMCallCompletedEvent, LLMCallStartedEvent, LLMCallType
    from crewai.events.types.task_events import TaskCompletedEvent, TaskFailedEvent, TaskStartedEvent
    from crewai.events.types.tool_usage_events import (
        ToolUsageErrorEvent,
        ToolUsageFinishedEvent,
        ToolUsageStartedEvent,
    )
except ImportError:

    class BaseEventListener:  # type: ignore[no-redef]
        pass

    CrewKickoffCompletedEvent = None  # type: ignore[assignment,misc]
    CrewKickoffFailedEvent = None  # type: ignore[assignment,misc]
    CrewKickoffStartedEvent = None  # type: ignore[assignment,misc]
    LLMCallCompletedEvent = None  # type: ignore[assignment,misc]
    LLMCallStartedEvent = None  # type: ignore[assignment,misc]
    LLMCallType = None  # type: ignore[assignment,misc]
    TaskCompletedEvent = None  # type: ignore[assignment,misc]
    TaskFailedEvent = None  # type: ignore[assignment,misc]
    TaskStartedEvent = None  # type: ignore[assignment,misc]
    ToolUsageErrorEvent = None  # type: ignore[assignment,misc]
    ToolUsageFinishedEvent = None  # type: ignore[assignment,misc]
    ToolUsageStartedEvent = None  # type: ignore[assignment,misc]

from stakeout_agent.backends.base import AbstractMonitorDB
from stakeout_agent.callback_handler.base import _MonitorBase

_ROLE_MAP = {"human": "human", "ai": "assistant", "system": "system", "tool": "tool"}


def _format_crewai_messages(messages: str | list[dict] | None, max_chars: int | None) -> list[dict]:
    """Convert CrewAI LLM messages to a flat list of {role, content} dicts."""
    if messages is None:
        return []
    if isinstance(messages, str):
        content = messages[:max_chars] if max_chars is not None else messages
        return [{"role": "user", "content": content}]
    result = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = _ROLE_MAP.get(str(m.get("role", "")), str(m.get("role", "user")))
        raw_content = m.get("content", "")
        if not isinstance(raw_content, str):
            raw_content = str(raw_content)
        content = raw_content[:max_chars] if max_chars is not None else raw_content
        result.append({"role": role, "content": content})
    return result


def _extract_crewai_response_text(response: Any) -> str | None:
    """Best-effort extraction of the response text from a CrewAI LLM response."""
    if response is None:
        return None
    if isinstance(response, str):
        return response
    # litellm / openai ModelResponse: choices[0].message.content
    try:
        return response.choices[0].message.content
    except (AttributeError, IndexError, TypeError):
        pass
    # Fallback
    try:
        return str(response)
    except Exception:
        return None


class CrewAIMonitorCallback(_MonitorBase, BaseEventListener):
    """Sync monitor for use with crew.kickoff().

    **Do not share a single instance across concurrent kickoffs or across multiple Crew
    objects.** Per-run state (_run_id, timing dicts) is stored on the instance and keyed
    by task/tool name; concurrent runs will overwrite each other's state and produce wrong
    latencies or events written under the wrong run ID. Create a new instance per
    crew.kickoff() call.

    Usage:
        monitor = CrewAIMonitorCallback(crew_id="my_crew", thread_id="thread_123")
        crew.kickoff(inputs={...})
    """

    def __init__(
        self,
        crew_id: str,
        thread_id: str,
        db: AbstractMonitorDB | None = None,
        capture_payloads: bool = True,
        max_payload_chars: int | None = None,
    ) -> None:
        if CrewKickoffStartedEvent is None:
            raise ImportError(
                "crewai is required for CrewAIMonitorCallback. Install it with: pip install 'stakeout-agent[crewai]'"
            )
        _MonitorBase.__init__(
            self, crew_id, thread_id, db, capture_payloads=capture_payloads, max_payload_chars=max_payload_chars
        )
        BaseEventListener.__init__(self)

    def setup_listeners(self, crewai_event_bus: Any) -> None:
        @crewai_event_bus.on(CrewKickoffStartedEvent)
        def on_crew_start(source: Any, event: CrewKickoffStartedEvent) -> None:
            run_id = str(uuid4())
            self._run_id = run_id
            self._safe_db_write(lambda: self.db.create_run(run_id, self.graph_id, self.thread_id))

        @crewai_event_bus.on(CrewKickoffCompletedEvent)
        def on_crew_end(source: Any, event: CrewKickoffCompletedEvent) -> None:
            run_id = self._run_id
            self._safe_db_write(lambda: self.db.complete_run(run_id))

        @crewai_event_bus.on(CrewKickoffFailedEvent)
        def on_crew_error(source: Any, event: CrewKickoffFailedEvent) -> None:
            run_id = self._run_id
            error_str = self._safe_truncate(event.error)
            self._safe_db_write(lambda: self.db.fail_run(run_id, error_str))

        @crewai_event_bus.on(TaskStartedEvent)
        def on_task_start(source: Any, event: TaskStartedEvent) -> None:
            task_name = event.task_name or "unknown_task"
            self._node_start_times[task_name] = time.monotonic()
            description = self._safe_truncate(getattr(event.task, "description", ""))
            run_id = self._run_id
            self._safe_db_write(
                lambda: self.db.insert_event(
                    run_id=run_id,
                    graph_id=self.graph_id,
                    event_type="node_start",
                    node_name=task_name,
                    payload={"description": description},
                )
            )

        @crewai_event_bus.on(LLMCallStartedEvent)
        def on_llm_start(source: Any, event: LLMCallStartedEvent) -> None:
            if not self.capture_payloads:
                return
            task_name = event.task_name or "unknown_task"
            formatted = _format_crewai_messages(event.messages, self._max_payload_chars)
            if formatted:
                with self._state_lock:
                    self._llm_inputs.setdefault(task_name, []).extend(formatted)

        @crewai_event_bus.on(LLMCallCompletedEvent)
        def on_llm_end(source: Any, event: LLMCallCompletedEvent) -> None:
            # Only capture text response for plain LLM calls, not tool-dispatch calls
            if not self.capture_payloads or event.call_type == LLMCallType.TOOL_CALL:
                return
            task_name = event.task_name or "unknown_task"
            text = _extract_crewai_response_text(event.response)
            if text is not None:
                if self._max_payload_chars is not None:
                    text = text[: self._max_payload_chars]
                with self._state_lock:
                    self._llm_outputs[task_name] = text

        @crewai_event_bus.on(TaskCompletedEvent)
        def on_task_end(source: Any, event: TaskCompletedEvent) -> None:
            task_name = event.task_name or "unknown_task"
            latency = self._pop_latency(self._node_start_times, task_name)
            with self._state_lock:
                llm_input = self._llm_inputs.pop(task_name, None) if self.capture_payloads else None
                llm_output = self._llm_outputs.pop(task_name, None) if self.capture_payloads else None
            run_id = self._run_id
            output = self._safe_truncate(event.output)
            self._safe_db_write(
                lambda: self.db.insert_event(
                    run_id=run_id,
                    graph_id=self.graph_id,
                    event_type="node_end",
                    node_name=task_name,
                    latency_ms=latency,
                    payload={"output": output},
                    llm_input=llm_input,
                    llm_output=llm_output,
                )
            )

        @crewai_event_bus.on(TaskFailedEvent)
        def on_task_error(source: Any, event: TaskFailedEvent) -> None:
            task_name = event.task_name or "unknown_task"
            latency = self._pop_latency(self._node_start_times, task_name)
            run_id = self._run_id
            error_str = self._safe_truncate(event.error)
            self._safe_db_write(
                lambda: self.db.insert_event(
                    run_id=run_id,
                    graph_id=self.graph_id,
                    event_type="error",
                    node_name=task_name,
                    latency_ms=latency,
                    error=error_str,
                )
            )

        @crewai_event_bus.on(ToolUsageStartedEvent)
        def on_tool_start(source: Any, event: ToolUsageStartedEvent) -> None:
            self._tool_start_times[event.tool_name] = time.monotonic()
            run_id = self._run_id
            tool_input = self._safe_truncate(event.tool_args)
            tool_name = event.tool_name
            self._safe_db_write(
                lambda: self.db.insert_event(
                    run_id=run_id,
                    graph_id=self.graph_id,
                    event_type="tool_call",
                    node_name=tool_name,
                    payload={"input": tool_input},
                )
            )

        @crewai_event_bus.on(ToolUsageFinishedEvent)
        def on_tool_end(source: Any, event: ToolUsageFinishedEvent) -> None:
            latency = self._pop_latency(self._tool_start_times, event.tool_name)
            run_id = self._run_id
            tool_output = self._safe_truncate(event.output)
            tool_name = event.tool_name
            self._safe_db_write(
                lambda: self.db.insert_event(
                    run_id=run_id,
                    graph_id=self.graph_id,
                    event_type="tool_result",
                    node_name=tool_name,
                    latency_ms=latency,
                    payload={"output": tool_output},
                )
            )

        @crewai_event_bus.on(ToolUsageErrorEvent)
        def on_tool_error(source: Any, event: ToolUsageErrorEvent) -> None:
            latency = self._pop_latency(self._tool_start_times, event.tool_name)
            run_id = self._run_id
            error_str = self._safe_truncate(event.error)
            tool_name = event.tool_name
            self._safe_db_write(
                lambda: self.db.insert_event(
                    run_id=run_id,
                    graph_id=self.graph_id,
                    event_type="error",
                    node_name=tool_name,
                    latency_ms=latency,
                    error=error_str,
                )
            )


class AsyncCrewAIMonitorCallback(_MonitorBase, BaseEventListener):
    """Async monitor for use with crew.kickoff_async() / crew.akickoff().

    Registers async event handlers so the CrewAI event bus routes them through
    its async pipeline. PyMongo calls are offloaded to a thread-pool executor to
    avoid blocking the event loop.

    **Do not share a single instance across concurrent kickoffs.** Per-run state
    (_run_id, timing dicts) is stored on the instance and keyed by task/tool name;
    concurrent async runs will overwrite each other's state. Create a new instance per
    crew.kickoff_async() call.

    Usage:
        monitor = AsyncCrewAIMonitorCallback(crew_id="my_crew", thread_id="thread_123")
        await crew.akickoff(inputs={...})
    """

    def __init__(
        self,
        crew_id: str,
        thread_id: str,
        db: AbstractMonitorDB | None = None,
        capture_payloads: bool = True,
        max_payload_chars: int | None = None,
    ) -> None:
        if CrewKickoffStartedEvent is None:
            raise ImportError(
                "crewai is required for AsyncCrewAIMonitorCallback. "
                "Install it with: pip install 'stakeout-agent[crewai]'"
            )
        _MonitorBase.__init__(
            self, crew_id, thread_id, db, capture_payloads=capture_payloads, max_payload_chars=max_payload_chars
        )
        BaseEventListener.__init__(self)

    def setup_listeners(self, crewai_event_bus: Any) -> None:
        @crewai_event_bus.on(CrewKickoffStartedEvent)
        async def on_crew_start(source: Any, event: CrewKickoffStartedEvent) -> None:
            run_id = str(uuid4())
            self._run_id = run_id
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None, lambda: self._safe_db_write(lambda: self.db.create_run(run_id, self.graph_id, self.thread_id))
            )

        @crewai_event_bus.on(CrewKickoffCompletedEvent)
        async def on_crew_end(source: Any, event: CrewKickoffCompletedEvent) -> None:
            run_id = self._run_id
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, lambda: self._safe_db_write(lambda: self.db.complete_run(run_id)))

        @crewai_event_bus.on(CrewKickoffFailedEvent)
        async def on_crew_error(source: Any, event: CrewKickoffFailedEvent) -> None:
            error_str = self._safe_truncate(event.error)
            run_id = self._run_id
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, lambda: self._safe_db_write(lambda: self.db.fail_run(run_id, error_str)))

        @crewai_event_bus.on(TaskStartedEvent)
        async def on_task_start(source: Any, event: TaskStartedEvent) -> None:
            task_name = event.task_name or "unknown_task"
            self._node_start_times[task_name] = time.monotonic()
            description = self._safe_truncate(getattr(event.task, "description", ""))
            run_id = self._run_id
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: self._safe_db_write(
                    lambda: self.db.insert_event(
                        run_id=run_id,
                        graph_id=self.graph_id,
                        event_type="node_start",
                        node_name=task_name,
                        payload={"description": description},
                    )
                ),
            )

        @crewai_event_bus.on(LLMCallStartedEvent)
        async def on_llm_start(source: Any, event: LLMCallStartedEvent) -> None:
            if not self.capture_payloads:
                return
            task_name = event.task_name or "unknown_task"
            formatted = _format_crewai_messages(event.messages, self._max_payload_chars)
            if formatted:
                with self._state_lock:
                    self._llm_inputs.setdefault(task_name, []).extend(formatted)

        @crewai_event_bus.on(LLMCallCompletedEvent)
        async def on_llm_end(source: Any, event: LLMCallCompletedEvent) -> None:
            # Only capture text response for plain LLM calls, not tool-dispatch calls
            if not self.capture_payloads or event.call_type == LLMCallType.TOOL_CALL:
                return
            task_name = event.task_name or "unknown_task"
            text = _extract_crewai_response_text(event.response)
            if text is not None:
                if self._max_payload_chars is not None:
                    text = text[: self._max_payload_chars]
                with self._state_lock:
                    self._llm_outputs[task_name] = text

        @crewai_event_bus.on(TaskCompletedEvent)
        async def on_task_end(source: Any, event: TaskCompletedEvent) -> None:
            task_name = event.task_name or "unknown_task"
            latency = self._pop_latency(self._node_start_times, task_name)
            output = self._safe_truncate(event.output)
            with self._state_lock:
                llm_input = self._llm_inputs.pop(task_name, None) if self.capture_payloads else None
                llm_output = self._llm_outputs.pop(task_name, None) if self.capture_payloads else None
            run_id = self._run_id
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: self._safe_db_write(
                    lambda: self.db.insert_event(
                        run_id=run_id,
                        graph_id=self.graph_id,
                        event_type="node_end",
                        node_name=task_name,
                        latency_ms=latency,
                        payload={"output": output},
                        llm_input=llm_input,
                        llm_output=llm_output,
                    )
                ),
            )

        @crewai_event_bus.on(TaskFailedEvent)
        async def on_task_error(source: Any, event: TaskFailedEvent) -> None:
            task_name = event.task_name or "unknown_task"
            latency = self._pop_latency(self._node_start_times, task_name)
            error_str = self._safe_truncate(event.error)
            run_id = self._run_id
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: self._safe_db_write(
                    lambda: self.db.insert_event(
                        run_id=run_id,
                        graph_id=self.graph_id,
                        event_type="error",
                        node_name=task_name,
                        latency_ms=latency,
                        error=error_str,
                    )
                ),
            )

        @crewai_event_bus.on(ToolUsageStartedEvent)
        async def on_tool_start(source: Any, event: ToolUsageStartedEvent) -> None:
            self._tool_start_times[event.tool_name] = time.monotonic()
            tool_input = self._safe_truncate(event.tool_args)
            tool_name = event.tool_name
            run_id = self._run_id
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: self._safe_db_write(
                    lambda: self.db.insert_event(
                        run_id=run_id,
                        graph_id=self.graph_id,
                        event_type="tool_call",
                        node_name=tool_name,
                        payload={"input": tool_input},
                    )
                ),
            )

        @crewai_event_bus.on(ToolUsageFinishedEvent)
        async def on_tool_end(source: Any, event: ToolUsageFinishedEvent) -> None:
            latency = self._pop_latency(self._tool_start_times, event.tool_name)
            output = self._safe_truncate(event.output)
            tool_name = event.tool_name
            run_id = self._run_id
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: self._safe_db_write(
                    lambda: self.db.insert_event(
                        run_id=run_id,
                        graph_id=self.graph_id,
                        event_type="tool_result",
                        node_name=tool_name,
                        latency_ms=latency,
                        payload={"output": output},
                    )
                ),
            )

        @crewai_event_bus.on(ToolUsageErrorEvent)
        async def on_tool_error(source: Any, event: ToolUsageErrorEvent) -> None:
            latency = self._pop_latency(self._tool_start_times, event.tool_name)
            error_str = self._safe_truncate(event.error)
            tool_name = event.tool_name
            run_id = self._run_id
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: self._safe_db_write(
                    lambda: self.db.insert_event(
                        run_id=run_id,
                        graph_id=self.graph_id,
                        event_type="error",
                        node_name=tool_name,
                        latency_ms=latency,
                        error=error_str,
                    )
                ),
            )
