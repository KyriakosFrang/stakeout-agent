from __future__ import annotations

import asyncio
import time
from typing import Any
from uuid import uuid4

from crewai.events.base_event_listener import BaseEventListener
from crewai.events.types.crew_events import (
    CrewKickoffCompletedEvent,
    CrewKickoffFailedEvent,
    CrewKickoffStartedEvent,
)
from crewai.events.types.task_events import TaskCompletedEvent, TaskFailedEvent, TaskStartedEvent
from crewai.events.types.tool_usage_events import (
    ToolUsageErrorEvent,
    ToolUsageFinishedEvent,
    ToolUsageStartedEvent,
)

from stakeout_agent.backends.base import AbstractMonitorDB
from stakeout_agent.callback_handler.base import _MonitorBase


class CrewAIMonitorCallback(_MonitorBase, BaseEventListener):
    """Sync monitor for use with crew.kickoff().

    Usage:
        monitor = CrewAIMonitorCallback(crew_id="my_crew", thread_id="thread_123")
        crew.kickoff(inputs={...})
    """

    def __init__(self, crew_id: str, thread_id: str, db: AbstractMonitorDB | None = None) -> None:
        _MonitorBase.__init__(self, crew_id, thread_id, db)
        BaseEventListener.__init__(self)

    def setup_listeners(self, crewai_event_bus: Any) -> None:
        @crewai_event_bus.on(CrewKickoffStartedEvent)
        def on_crew_start(source: Any, event: CrewKickoffStartedEvent) -> None:
            run_id = str(uuid4())
            self._run_id = run_id
            self.db.create_run(run_id, self.graph_id, self.thread_id)

        @crewai_event_bus.on(CrewKickoffCompletedEvent)
        def on_crew_end(source: Any, event: CrewKickoffCompletedEvent) -> None:
            self.db.complete_run(self._run_id)

        @crewai_event_bus.on(CrewKickoffFailedEvent)
        def on_crew_error(source: Any, event: CrewKickoffFailedEvent) -> None:
            self.db.fail_run(self._run_id, self._safe_truncate(event.error))

        @crewai_event_bus.on(TaskStartedEvent)
        def on_task_start(source: Any, event: TaskStartedEvent) -> None:
            task_name = event.task_name or "unknown_task"
            self._node_start_times[task_name] = time.monotonic()
            self.db.insert_event(
                run_id=self._run_id,
                graph_id=self.graph_id,
                event_type="node_start",
                node_name=task_name,
                payload={"description": self._safe_truncate(getattr(event.task, "description", ""))},
            )

        @crewai_event_bus.on(TaskCompletedEvent)
        def on_task_end(source: Any, event: TaskCompletedEvent) -> None:
            task_name = event.task_name or "unknown_task"
            latency = self._pop_latency(self._node_start_times, task_name)
            self.db.insert_event(
                run_id=self._run_id,
                graph_id=self.graph_id,
                event_type="node_end",
                node_name=task_name,
                latency_ms=latency,
                payload={"output": self._safe_truncate(event.output)},
            )

        @crewai_event_bus.on(TaskFailedEvent)
        def on_task_error(source: Any, event: TaskFailedEvent) -> None:
            task_name = event.task_name or "unknown_task"
            latency = self._pop_latency(self._node_start_times, task_name)
            self.db.insert_event(
                run_id=self._run_id,
                graph_id=self.graph_id,
                event_type="error",
                node_name=task_name,
                latency_ms=latency,
                error=self._safe_truncate(event.error),
            )

        @crewai_event_bus.on(ToolUsageStartedEvent)
        def on_tool_start(source: Any, event: ToolUsageStartedEvent) -> None:
            self._tool_start_times[event.tool_name] = time.monotonic()
            self.db.insert_event(
                run_id=self._run_id,
                graph_id=self.graph_id,
                event_type="tool_call",
                node_name=event.tool_name,
                payload={"input": self._safe_truncate(event.tool_args)},
            )

        @crewai_event_bus.on(ToolUsageFinishedEvent)
        def on_tool_end(source: Any, event: ToolUsageFinishedEvent) -> None:
            latency = self._pop_latency(self._tool_start_times, event.tool_name)
            self.db.insert_event(
                run_id=self._run_id,
                graph_id=self.graph_id,
                event_type="tool_result",
                node_name=event.tool_name,
                latency_ms=latency,
                payload={"output": self._safe_truncate(event.output)},
            )

        @crewai_event_bus.on(ToolUsageErrorEvent)
        def on_tool_error(source: Any, event: ToolUsageErrorEvent) -> None:
            latency = self._pop_latency(self._tool_start_times, event.tool_name)
            self.db.insert_event(
                run_id=self._run_id,
                graph_id=self.graph_id,
                event_type="error",
                node_name=event.tool_name,
                latency_ms=latency,
                error=self._safe_truncate(event.error),
            )


class AsyncCrewAIMonitorCallback(_MonitorBase, BaseEventListener):
    """Async monitor for use with crew.kickoff_async() / crew.akickoff().

    Registers async event handlers so the CrewAI event bus routes them through
    its async pipeline. PyMongo calls are offloaded to a thread-pool executor to
    avoid blocking the event loop.

    Usage:
        monitor = AsyncCrewAIMonitorCallback(crew_id="my_crew", thread_id="thread_123")
        await crew.akickoff(inputs={...})
    """

    def __init__(self, crew_id: str, thread_id: str, db: AbstractMonitorDB | None = None) -> None:
        _MonitorBase.__init__(self, crew_id, thread_id, db)
        BaseEventListener.__init__(self)

    def setup_listeners(self, crewai_event_bus: Any) -> None:
        @crewai_event_bus.on(CrewKickoffStartedEvent)
        async def on_crew_start(source: Any, event: CrewKickoffStartedEvent) -> None:
            run_id = str(uuid4())
            self._run_id = run_id
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, lambda: self.db.create_run(run_id, self.graph_id, self.thread_id))

        @crewai_event_bus.on(CrewKickoffCompletedEvent)
        async def on_crew_end(source: Any, event: CrewKickoffCompletedEvent) -> None:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, lambda: self.db.complete_run(self._run_id))

        @crewai_event_bus.on(CrewKickoffFailedEvent)
        async def on_crew_error(source: Any, event: CrewKickoffFailedEvent) -> None:
            error_str = self._safe_truncate(event.error)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, lambda: self.db.fail_run(self._run_id, error_str))

        @crewai_event_bus.on(TaskStartedEvent)
        async def on_task_start(source: Any, event: TaskStartedEvent) -> None:
            task_name = event.task_name or "unknown_task"
            self._node_start_times[task_name] = time.monotonic()
            description = self._safe_truncate(getattr(event.task, "description", ""))
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: self.db.insert_event(
                    run_id=self._run_id,
                    graph_id=self.graph_id,
                    event_type="node_start",
                    node_name=task_name,
                    payload={"description": description},
                ),
            )

        @crewai_event_bus.on(TaskCompletedEvent)
        async def on_task_end(source: Any, event: TaskCompletedEvent) -> None:
            task_name = event.task_name or "unknown_task"
            latency = self._pop_latency(self._node_start_times, task_name)
            output = self._safe_truncate(event.output)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: self.db.insert_event(
                    run_id=self._run_id,
                    graph_id=self.graph_id,
                    event_type="node_end",
                    node_name=task_name,
                    latency_ms=latency,
                    payload={"output": output},
                ),
            )

        @crewai_event_bus.on(TaskFailedEvent)
        async def on_task_error(source: Any, event: TaskFailedEvent) -> None:
            task_name = event.task_name or "unknown_task"
            latency = self._pop_latency(self._node_start_times, task_name)
            error_str = self._safe_truncate(event.error)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: self.db.insert_event(
                    run_id=self._run_id,
                    graph_id=self.graph_id,
                    event_type="error",
                    node_name=task_name,
                    latency_ms=latency,
                    error=error_str,
                ),
            )

        @crewai_event_bus.on(ToolUsageStartedEvent)
        async def on_tool_start(source: Any, event: ToolUsageStartedEvent) -> None:
            self._tool_start_times[event.tool_name] = time.monotonic()
            tool_input = self._safe_truncate(event.tool_args)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: self.db.insert_event(
                    run_id=self._run_id,
                    graph_id=self.graph_id,
                    event_type="tool_call",
                    node_name=event.tool_name,
                    payload={"input": tool_input},
                ),
            )

        @crewai_event_bus.on(ToolUsageFinishedEvent)
        async def on_tool_end(source: Any, event: ToolUsageFinishedEvent) -> None:
            latency = self._pop_latency(self._tool_start_times, event.tool_name)
            output = self._safe_truncate(event.output)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: self.db.insert_event(
                    run_id=self._run_id,
                    graph_id=self.graph_id,
                    event_type="tool_result",
                    node_name=event.tool_name,
                    latency_ms=latency,
                    payload={"output": output},
                ),
            )

        @crewai_event_bus.on(ToolUsageErrorEvent)
        async def on_tool_error(source: Any, event: ToolUsageErrorEvent) -> None:
            latency = self._pop_latency(self._tool_start_times, event.tool_name)
            error_str = self._safe_truncate(event.error)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: self.db.insert_event(
                    run_id=self._run_id,
                    graph_id=self.graph_id,
                    event_type="error",
                    node_name=event.tool_name,
                    latency_ms=latency,
                    error=error_str,
                ),
            )
