"""In-process OTEL backend tests using InMemorySpanExporter.

No Docker required — these always run as part of the normal test suite.
They verify actual span content, parent-child structure, attributes, and events
rather than just mock call invocations.
"""

from __future__ import annotations

import json

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from stakeout_agent.backends.otel import OTELMonitorDB


def _make_db():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    db = OTELMonitorDB(tracer_provider=provider)
    return db, exporter


# ---------------------------------------------------------------------------
# Full happy-path lifecycle
# ---------------------------------------------------------------------------


class TestFullLifecycle:
    def test_span_count_and_names(self):
        db, exporter = _make_db()
        db.create_run("run-1", "my_graph", "thread-42")
        db.insert_event(run_id="run-1", graph_id="my_graph", event_type="node_start", node_name="agent")
        db.insert_event(run_id="run-1", graph_id="my_graph", event_type="node_end", node_name="agent")
        db.insert_event(run_id="run-1", graph_id="my_graph", event_type="tool_call", node_name="search")
        db.insert_event(run_id="run-1", graph_id="my_graph", event_type="tool_result", node_name="search")
        db.complete_run("run-1")

        spans = exporter.get_finished_spans()
        assert len(spans) == 3
        names = {s.name for s in spans}
        assert names == {"my_graph", "agent", "search"}

    def test_parent_child_relationships(self):
        db, exporter = _make_db()
        db.create_run("run-1", "my_graph", "thread-42")
        db.insert_event(run_id="run-1", graph_id="my_graph", event_type="node_start", node_name="agent")
        db.insert_event(run_id="run-1", graph_id="my_graph", event_type="node_end", node_name="agent")
        db.insert_event(run_id="run-1", graph_id="my_graph", event_type="tool_call", node_name="search")
        db.insert_event(run_id="run-1", graph_id="my_graph", event_type="tool_result", node_name="search")
        db.complete_run("run-1")

        by_name = {s.name: s for s in exporter.get_finished_spans()}
        root = by_name["my_graph"]
        node = by_name["agent"]
        tool = by_name["search"]

        assert root.parent is None
        assert node.parent.span_id == root.context.span_id
        assert tool.parent.span_id == root.context.span_id

    def test_root_span_stakeout_attributes(self):
        db, exporter = _make_db()
        db.create_run("run-1", "my_graph", "thread-42")
        db.complete_run("run-1", total_input_tokens=100, total_output_tokens=50, estimated_cost_usd=0.002)

        root = exporter.get_finished_spans()[0]
        assert root.attributes["stakeout.run_id"] == "run-1"
        assert root.attributes["stakeout.graph_id"] == "my_graph"
        assert root.attributes["stakeout.thread_id"] == "thread-42"
        assert root.attributes["gen_ai.usage.input_tokens"] == 100
        assert root.attributes["gen_ai.usage.output_tokens"] == 50
        assert root.attributes["stakeout.cost_usd"] == pytest.approx(0.002)

    def test_root_span_cache_token_attributes(self):
        db, exporter = _make_db()
        db.create_run("run-1", "g", "t")
        db.complete_run("run-1", total_cache_read_tokens=30, total_cache_creation_tokens=10)

        root = exporter.get_finished_spans()[0]
        assert root.attributes["gen_ai.usage.cache_read_input_tokens"] == 30
        assert root.attributes["gen_ai.usage.cache_creation_input_tokens"] == 10

    def test_complete_run_sets_ok_status(self):
        db, exporter = _make_db()
        db.create_run("run-1", "g", "t")
        db.complete_run("run-1")

        root = exporter.get_finished_spans()[0]
        assert root.status.status_code == StatusCode.OK

    def test_node_span_attributes(self):
        db, exporter = _make_db()
        db.create_run("run-1", "g", "t")
        db.insert_event(run_id="run-1", graph_id="g", event_type="node_start", node_name="agent")
        db.insert_event(
            run_id="run-1",
            graph_id="g",
            event_type="node_end",
            node_name="agent",
            latency_ms=150.0,
            input_tokens=60,
            output_tokens=25,
            model="gpt-4o",
            cache_read_tokens=15,
            cache_creation_tokens=3,
        )
        db.complete_run("run-1")

        by_name = {s.name: s for s in exporter.get_finished_spans()}
        node = by_name["agent"]
        assert node.attributes["stakeout.latency_ms"] == pytest.approx(150.0)
        assert node.attributes["gen_ai.usage.input_tokens"] == 60
        assert node.attributes["gen_ai.usage.output_tokens"] == 25
        assert node.attributes["gen_ai.request.model"] == "gpt-4o"
        assert node.attributes["gen_ai.usage.cache_read_input_tokens"] == 15
        assert node.attributes["gen_ai.usage.cache_creation_input_tokens"] == 3
        assert node.status.status_code == StatusCode.OK

    def test_tool_span_latency_attribute(self):
        db, exporter = _make_db()
        db.create_run("run-1", "g", "t")
        db.insert_event(run_id="run-1", graph_id="g", event_type="tool_call", node_name="calculator")
        db.insert_event(run_id="run-1", graph_id="g", event_type="tool_result", node_name="calculator", latency_ms=8.5)
        db.complete_run("run-1")

        by_name = {s.name: s for s in exporter.get_finished_spans()}
        tool = by_name["calculator"]
        assert tool.attributes["stakeout.latency_ms"] == pytest.approx(8.5)
        assert tool.status.status_code == StatusCode.OK


# ---------------------------------------------------------------------------
# LLM I/O as span events, not attributes
# ---------------------------------------------------------------------------


class TestLlmIoAsSpanEvents:
    def test_llm_input_output_become_span_events(self):
        db, exporter = _make_db()
        db.create_run("run-1", "g", "t")
        db.insert_event(run_id="run-1", graph_id="g", event_type="node_start", node_name="agent")
        db.insert_event(
            run_id="run-1",
            graph_id="g",
            event_type="node_end",
            node_name="agent",
            llm_input=[{"role": "user", "content": "hello"}],
            llm_output="world",
        )
        db.complete_run("run-1")

        by_name = {s.name: s for s in exporter.get_finished_spans()}
        node = by_name["agent"]
        event_names = [e.name for e in node.events]
        assert "gen_ai.content.prompt" in event_names
        assert "gen_ai.content.completion" in event_names

    def test_llm_input_body_is_json_string(self):
        db, exporter = _make_db()
        db.create_run("run-1", "g", "t")
        db.insert_event(run_id="run-1", graph_id="g", event_type="node_start", node_name="n")
        db.insert_event(
            run_id="run-1",
            graph_id="g",
            event_type="node_end",
            node_name="n",
            llm_input=[{"role": "system", "content": "You are helpful."}],
            llm_output="sure",
        )
        db.complete_run("run-1")

        by_name = {s.name: s for s in exporter.get_finished_spans()}
        node = by_name["n"]
        prompt_event = next(e for e in node.events if e.name == "gen_ai.content.prompt")
        body = json.loads(prompt_event.attributes["body"])
        assert body[0]["role"] == "system"

    def test_llm_io_not_stored_as_span_attributes(self):
        db, exporter = _make_db()
        db.create_run("run-1", "g", "t")
        db.insert_event(run_id="run-1", graph_id="g", event_type="node_start", node_name="n")
        db.insert_event(
            run_id="run-1",
            graph_id="g",
            event_type="node_end",
            node_name="n",
            llm_input=[{"role": "user", "content": "hi"}],
            llm_output="hello",
        )
        db.complete_run("run-1")

        by_name = {s.name: s for s in exporter.get_finished_spans()}
        node = by_name["n"]
        assert "llm_input" not in node.attributes
        assert "llm_output" not in node.attributes


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestErrorPaths:
    def test_fail_run_sets_error_status(self):
        db, exporter = _make_db()
        db.create_run("run-1", "g", "t")
        db.fail_run("run-1", "RuntimeError: crash")

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        root = spans[0]
        assert root.status.status_code == StatusCode.ERROR
        assert root.status.description == "RuntimeError: crash"

    def test_fail_run_records_exception_event(self):
        db, exporter = _make_db()
        db.create_run("run-1", "g", "t")
        db.fail_run("run-1", "RuntimeError: crash")

        root = exporter.get_finished_spans()[0]
        assert any(e.name == "exception" for e in root.events)

    def test_node_error_sets_error_status_on_node_span(self):
        db, exporter = _make_db()
        db.create_run("run-1", "g", "t")
        db.insert_event(run_id="run-1", graph_id="g", event_type="node_start", node_name="agent")
        db.insert_event(run_id="run-1", graph_id="g", event_type="error", node_name="agent", error="ValueError: bad")
        db.complete_run("run-1")

        by_name = {s.name: s for s in exporter.get_finished_spans()}
        assert "agent" in by_name
        assert by_name["agent"].status.status_code == StatusCode.ERROR

    def test_tool_error_sets_error_status_on_tool_span(self):
        db, exporter = _make_db()
        db.create_run("run-1", "g", "t")
        db.insert_event(run_id="run-1", graph_id="g", event_type="tool_call", node_name="search")
        db.insert_event(run_id="run-1", graph_id="g", event_type="error", node_name="search", error="ConnectionError")
        db.complete_run("run-1")

        by_name = {s.name: s for s in exporter.get_finished_spans()}
        assert "search" in by_name
        assert by_name["search"].status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# Retriever lifecycle
# ---------------------------------------------------------------------------


class TestRetrieverLifecycle:
    def test_retriever_spans_parented_under_root(self):
        db, exporter = _make_db()
        db.create_run("run-1", "g", "t")
        db.insert_event(run_id="run-1", graph_id="g", event_type="retriever_start", node_name="vector_store")
        db.insert_event(
            run_id="run-1", graph_id="g", event_type="retriever_end", node_name="vector_store", latency_ms=12.0
        )
        db.complete_run("run-1")

        by_name = {s.name: s for s in exporter.get_finished_spans()}
        assert "vector_store" in by_name
        root = by_name["g"]
        retriever = by_name["vector_store"]
        assert retriever.parent.span_id == root.context.span_id
        assert retriever.attributes["stakeout.latency_ms"] == pytest.approx(12.0)
