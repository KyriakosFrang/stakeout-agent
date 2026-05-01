from __future__ import annotations

import time
from datetime import datetime, timezone
from unittest.mock import MagicMock

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
from crewai.tasks.task_output import TaskOutput

from stakeout_agent.callback_handler.crewai import AsyncCrewAIMonitorCallback, CrewAIMonitorCallback

CREW_ID = "test_crew"
THREAD_ID = "thread_1"


class MockBus:
    def __init__(self):
        self._handlers: dict = {}

    def on(self, event_type):
        def decorator(fn):
            self._handlers[event_type] = fn
            return fn

        return decorator

    def validate_dependencies(self):
        pass

    def emit(self, event_type, source, event):
        if event_type in self._handlers:
            self._handlers[event_type](source, event)

    async def aemit(self, event_type, source, event):
        if event_type in self._handlers:
            await self._handlers[event_type](source, event)


def _make() -> tuple[CrewAIMonitorCallback, MagicMock, MockBus]:
    db = MagicMock()
    bus = MockBus()

    import crewai.events.base_event_listener as _mod

    original_bus = _mod.crewai_event_bus
    _mod.crewai_event_bus = bus
    try:
        cb = CrewAIMonitorCallback(crew_id=CREW_ID, thread_id=THREAD_ID, db=db)
    finally:
        _mod.crewai_event_bus = original_bus

    return cb, db, bus


def _make_async() -> tuple[AsyncCrewAIMonitorCallback, MagicMock, MockBus]:
    db = MagicMock()
    bus = MockBus()

    import crewai.events.base_event_listener as _mod

    original_bus = _mod.crewai_event_bus
    _mod.crewai_event_bus = bus
    try:
        cb = AsyncCrewAIMonitorCallback(crew_id=CREW_ID, thread_id=THREAD_ID, db=db)
    finally:
        _mod.crewai_event_bus = original_bus

    return cb, db, bus


def _crew_started_event(**kwargs) -> CrewKickoffStartedEvent:
    return CrewKickoffStartedEvent(crew_name=CREW_ID, crew=None, inputs={}, **kwargs)


def _crew_completed_event(**kwargs) -> CrewKickoffCompletedEvent:
    return CrewKickoffCompletedEvent(crew_name=CREW_ID, crew=None, output=MagicMock(), total_tokens=0, **kwargs)


def _crew_failed_event(error: str = "Boom") -> CrewKickoffFailedEvent:
    return CrewKickoffFailedEvent(crew_name=CREW_ID, crew=None, error=error)


def _task_output(raw: str = "Done") -> TaskOutput:
    return TaskOutput(description="task", raw=raw, agent="test_agent")


def _task_started_event(task_name: str = "write report") -> TaskStartedEvent:
    # task=None prevents _set_task_fingerprint from overwriting task_name with mock attributes
    return TaskStartedEvent(task_name=task_name, task=None, context=None)


def _task_completed_event(task_name: str = "write report", raw: str = "Done") -> TaskCompletedEvent:
    return TaskCompletedEvent(task_name=task_name, task=None, output=_task_output(raw))


def _task_failed_event(task_name: str = "write report", error: str = "Failed") -> TaskFailedEvent:
    return TaskFailedEvent(task_name=task_name, task=None, error=error)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _tool_started_event(tool_name: str = "search", tool_args: str = "query") -> ToolUsageStartedEvent:
    return ToolUsageStartedEvent(
        tool_name=tool_name,
        tool_args=tool_args,
        tool_class=None,
        agent=None,
        from_task=None,
        from_agent=None,
    )


def _tool_finished_event(tool_name: str = "search", output: str = "results") -> ToolUsageFinishedEvent:
    return ToolUsageFinishedEvent(
        tool_name=tool_name,
        tool_args="query",
        tool_class=None,
        agent=None,
        from_task=None,
        from_agent=None,
        started_at=_now(),
        finished_at=_now(),
        from_cache=False,
        output=output,
    )


def _tool_error_event(tool_name: str = "search", error: str = "Tool failed") -> ToolUsageErrorEvent:
    return ToolUsageErrorEvent(
        tool_name=tool_name,
        tool_args="query",
        tool_class=None,
        agent=None,
        from_task=None,
        from_agent=None,
        error=error,
    )


# ---------------------------------------------------------------------------
# Crew lifecycle
# ---------------------------------------------------------------------------


class TestCrewLifecycle:
    def test_crew_start_creates_run(self):
        cb, db, bus = _make()
        bus.emit(CrewKickoffStartedEvent, None, _crew_started_event())
        db.create_run.assert_called_once()
        args = db.create_run.call_args.args
        assert args[1] == CREW_ID
        assert args[2] == THREAD_ID
        assert cb._run_id is not None

    def test_crew_start_stores_run_id(self):
        cb, db, bus = _make()
        bus.emit(CrewKickoffStartedEvent, None, _crew_started_event())
        run_id = cb._run_id
        assert isinstance(run_id, str) and len(run_id) == 36  # UUID format

    def test_crew_end_completes_run(self):
        cb, db, bus = _make()
        bus.emit(CrewKickoffStartedEvent, None, _crew_started_event())
        bus.emit(CrewKickoffCompletedEvent, None, _crew_completed_event())
        db.complete_run.assert_called_once_with(cb._run_id)

    def test_crew_error_fails_run(self):
        cb, db, bus = _make()
        bus.emit(CrewKickoffStartedEvent, None, _crew_started_event())
        bus.emit(CrewKickoffFailedEvent, None, _crew_failed_event(error="Something went wrong"))
        db.fail_run.assert_called_once_with(cb._run_id, "Something went wrong")


# ---------------------------------------------------------------------------
# Task events
# ---------------------------------------------------------------------------


class TestTaskEvents:
    def test_task_start_inserts_node_start_event(self):
        cb, db, bus = _make()
        bus.emit(CrewKickoffStartedEvent, None, _crew_started_event())
        bus.emit(TaskStartedEvent, None, _task_started_event(task_name="analyse data"))
        db.insert_event.assert_called_once()
        kwargs = db.insert_event.call_args.kwargs
        assert kwargs["event_type"] == "node_start"
        assert kwargs["node_name"] == "analyse data"
        assert kwargs["run_id"] == cb._run_id
        assert kwargs["graph_id"] == CREW_ID

    def test_task_end_inserts_node_end_event_with_latency(self):
        cb, db, bus = _make()
        bus.emit(CrewKickoffStartedEvent, None, _crew_started_event())
        cb._node_start_times["write report"] = time.monotonic() - 0.05
        bus.emit(TaskCompletedEvent, None, _task_completed_event())
        kwargs = db.insert_event.call_args.kwargs
        assert kwargs["event_type"] == "node_end"
        assert kwargs["node_name"] == "write report"
        assert kwargs["latency_ms"] is not None and kwargs["latency_ms"] > 0

    def test_task_end_includes_output_in_payload(self):
        cb, db, bus = _make()
        bus.emit(CrewKickoffStartedEvent, None, _crew_started_event())
        cb._node_start_times["write report"] = time.monotonic()
        bus.emit(TaskCompletedEvent, None, _task_completed_event(raw="Final report"))
        kwargs = db.insert_event.call_args.kwargs
        assert "output" in kwargs["payload"]

    def test_task_error_inserts_error_event(self):
        cb, db, bus = _make()
        bus.emit(CrewKickoffStartedEvent, None, _crew_started_event())
        cb._node_start_times["write report"] = time.monotonic()
        bus.emit(TaskFailedEvent, None, _task_failed_event(error="LLM timeout"))
        kwargs = db.insert_event.call_args.kwargs
        assert kwargs["event_type"] == "error"
        assert kwargs["error"] == "LLM timeout"
        assert kwargs["node_name"] == "write report"

    def test_task_error_latency_is_none_without_start(self):
        cb, db, bus = _make()
        bus.emit(CrewKickoffStartedEvent, None, _crew_started_event())
        bus.emit(TaskFailedEvent, None, _task_failed_event())
        kwargs = db.insert_event.call_args.kwargs
        assert kwargs["latency_ms"] is None


# ---------------------------------------------------------------------------
# Tool events
# ---------------------------------------------------------------------------


class TestToolEvents:
    def test_tool_start_inserts_tool_call_event(self):
        cb, db, bus = _make()
        bus.emit(CrewKickoffStartedEvent, None, _crew_started_event())
        bus.emit(ToolUsageStartedEvent, None, _tool_started_event(tool_name="web_search", tool_args="AI news"))
        kwargs = db.insert_event.call_args.kwargs
        assert kwargs["event_type"] == "tool_call"
        assert kwargs["node_name"] == "web_search"
        assert kwargs["payload"]["input"] == "AI news"

    def test_tool_end_inserts_tool_result_event_with_latency(self):
        cb, db, bus = _make()
        bus.emit(CrewKickoffStartedEvent, None, _crew_started_event())
        cb._tool_start_times["web_search"] = time.monotonic() - 0.05
        bus.emit(ToolUsageFinishedEvent, None, _tool_finished_event(tool_name="web_search", output="Latest AI news"))
        kwargs = db.insert_event.call_args.kwargs
        assert kwargs["event_type"] == "tool_result"
        assert kwargs["node_name"] == "web_search"
        assert kwargs["latency_ms"] is not None and kwargs["latency_ms"] > 0
        assert kwargs["payload"]["output"] == "Latest AI news"

    def test_tool_error_inserts_error_event(self):
        cb, db, bus = _make()
        bus.emit(CrewKickoffStartedEvent, None, _crew_started_event())
        cb._tool_start_times["web_search"] = time.monotonic()
        bus.emit(ToolUsageErrorEvent, None, _tool_error_event(tool_name="web_search", error="Connection refused"))
        kwargs = db.insert_event.call_args.kwargs
        assert kwargs["event_type"] == "error"
        assert kwargs["node_name"] == "web_search"
        assert kwargs["error"] == "Connection refused"

    def test_tool_error_latency_is_none_without_start(self):
        cb, db, bus = _make()
        bus.emit(CrewKickoffStartedEvent, None, _crew_started_event())
        bus.emit(ToolUsageErrorEvent, None, _tool_error_event())
        kwargs = db.insert_event.call_args.kwargs
        assert kwargs["latency_ms"] is None


# ---------------------------------------------------------------------------
# AsyncCrewAIMonitorCallback
# ---------------------------------------------------------------------------


class TestAsyncCrewAICallback:
    async def test_crew_start_creates_run(self):
        cb, db, bus = _make_async()
        await bus.aemit(CrewKickoffStartedEvent, None, _crew_started_event())
        db.create_run.assert_called_once()
        args = db.create_run.call_args.args
        assert args[1] == CREW_ID
        assert args[2] == THREAD_ID
        assert cb._run_id is not None

    async def test_crew_end_completes_run(self):
        cb, db, bus = _make_async()
        await bus.aemit(CrewKickoffStartedEvent, None, _crew_started_event())
        await bus.aemit(CrewKickoffCompletedEvent, None, _crew_completed_event())
        db.complete_run.assert_called_once_with(cb._run_id)

    async def test_crew_error_fails_run(self):
        cb, db, bus = _make_async()
        await bus.aemit(CrewKickoffStartedEvent, None, _crew_started_event())
        await bus.aemit(CrewKickoffFailedEvent, None, _crew_failed_event(error="Async boom"))
        db.fail_run.assert_called_once_with(cb._run_id, "Async boom")

    async def test_task_start_inserts_node_start_event(self):
        cb, db, bus = _make_async()
        await bus.aemit(CrewKickoffStartedEvent, None, _crew_started_event())
        await bus.aemit(TaskStartedEvent, None, _task_started_event(task_name="analyse data"))
        kwargs = db.insert_event.call_args.kwargs
        assert kwargs["event_type"] == "node_start"
        assert kwargs["node_name"] == "analyse data"

    async def test_task_end_inserts_node_end_with_latency(self):
        cb, db, bus = _make_async()
        await bus.aemit(CrewKickoffStartedEvent, None, _crew_started_event())
        cb._node_start_times["write report"] = time.monotonic() - 0.05
        await bus.aemit(TaskCompletedEvent, None, _task_completed_event())
        kwargs = db.insert_event.call_args.kwargs
        assert kwargs["event_type"] == "node_end"
        assert kwargs["latency_ms"] is not None and kwargs["latency_ms"] > 0

    async def test_task_error_inserts_error_event(self):
        cb, db, bus = _make_async()
        await bus.aemit(CrewKickoffStartedEvent, None, _crew_started_event())
        cb._node_start_times["write report"] = time.monotonic()
        await bus.aemit(TaskFailedEvent, None, _task_failed_event(error="Async timeout"))
        kwargs = db.insert_event.call_args.kwargs
        assert kwargs["event_type"] == "error"
        assert kwargs["error"] == "Async timeout"

    async def test_tool_start_inserts_tool_call_event(self):
        cb, db, bus = _make_async()
        await bus.aemit(CrewKickoffStartedEvent, None, _crew_started_event())
        await bus.aemit(ToolUsageStartedEvent, None, _tool_started_event(tool_name="web_search", tool_args="AI news"))
        kwargs = db.insert_event.call_args.kwargs
        assert kwargs["event_type"] == "tool_call"
        assert kwargs["node_name"] == "web_search"

    async def test_tool_end_inserts_tool_result_with_latency(self):
        cb, db, bus = _make_async()
        await bus.aemit(CrewKickoffStartedEvent, None, _crew_started_event())
        cb._tool_start_times["web_search"] = time.monotonic() - 0.05
        await bus.aemit(ToolUsageFinishedEvent, None, _tool_finished_event(tool_name="web_search"))
        kwargs = db.insert_event.call_args.kwargs
        assert kwargs["event_type"] == "tool_result"
        assert kwargs["latency_ms"] is not None and kwargs["latency_ms"] > 0

    async def test_tool_error_inserts_error_event(self):
        cb, db, bus = _make_async()
        await bus.aemit(CrewKickoffStartedEvent, None, _crew_started_event())
        cb._tool_start_times["web_search"] = time.monotonic()
        await bus.aemit(ToolUsageErrorEvent, None, _tool_error_event(tool_name="web_search", error="Async refused"))
        kwargs = db.insert_event.call_args.kwargs
        assert kwargs["event_type"] == "error"
        assert kwargs["error"] == "Async refused"
