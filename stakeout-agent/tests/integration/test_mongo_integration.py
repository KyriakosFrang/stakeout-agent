"""Integration tests for MongoMonitorDB against a real MongoDB instance.

Requires: docker compose up -d mongo
Skipped automatically when MongoDB is not reachable.
"""

from __future__ import annotations

import pytest


@pytest.mark.integration
class TestMongoCreateRun:
    def test_run_document_created_with_running_status(self, mongo_db, run_id):
        try:
            mongo_db.create_run(run_id, "test_graph", "test_thread")

            doc = mongo_db.runs.find_one({"_id": run_id})
            assert doc is not None
            assert doc["status"] == "running"
            assert doc["graph_id"] == "test_graph"
            assert doc["thread_id"] == "test_thread"
            assert doc["ended_at"] is None
            assert doc["error"] is None
        finally:
            mongo_db.runs.delete_one({"_id": run_id})
            mongo_db.events.delete_many({"run_id": run_id})


@pytest.mark.integration
class TestMongoCompleteRun:
    def test_run_marked_completed_with_token_fields(self, mongo_db, run_id):
        try:
            mongo_db.create_run(run_id, "g", "t")
            mongo_db.complete_run(
                run_id,
                total_input_tokens=120,
                total_output_tokens=40,
                estimated_cost_usd=0.001,
                total_cache_read_tokens=10,
                total_cache_creation_tokens=5,
            )

            doc = mongo_db.runs.find_one({"_id": run_id})
            assert doc["status"] == "completed"
            assert doc["ended_at"] is not None
            assert doc["total_input_tokens"] == 120
            assert doc["total_output_tokens"] == 40
            assert doc["estimated_cost_usd"] == pytest.approx(0.001)
            assert doc["total_cache_read_tokens"] == 10
            assert doc["total_cache_creation_tokens"] == 5
        finally:
            mongo_db.runs.delete_one({"_id": run_id})
            mongo_db.events.delete_many({"run_id": run_id})


@pytest.mark.integration
class TestMongoFailRun:
    def test_run_marked_failed_with_error(self, mongo_db, run_id):
        try:
            mongo_db.create_run(run_id, "g", "t")
            mongo_db.fail_run(run_id, "ValueError: something broke")

            doc = mongo_db.runs.find_one({"_id": run_id})
            assert doc["status"] == "failed"
            assert doc["ended_at"] is not None
            assert doc["error"] == "ValueError: something broke"
        finally:
            mongo_db.runs.delete_one({"_id": run_id})
            mongo_db.events.delete_many({"run_id": run_id})


@pytest.mark.integration
class TestMongoInsertEvent:
    def test_node_start_event_stored(self, mongo_db, run_id):
        try:
            mongo_db.create_run(run_id, "g", "t")
            mongo_db.insert_event(
                run_id=run_id,
                graph_id="g",
                event_type="node_start",
                node_name="agent",
                payload={"inputs": "hello"},
            )

            event = mongo_db.events.find_one({"run_id": run_id, "event_type": "node_start"})
            assert event is not None
            assert event["node_name"] == "agent"
            assert event["payload"] == {"inputs": "hello"}
        finally:
            mongo_db.runs.delete_one({"_id": run_id})
            mongo_db.events.delete_many({"run_id": run_id})

    def test_node_end_event_stores_tokens_and_model(self, mongo_db, run_id):
        try:
            mongo_db.create_run(run_id, "g", "t")
            mongo_db.insert_event(
                run_id=run_id,
                graph_id="g",
                event_type="node_end",
                node_name="agent",
                latency_ms=250.5,
                input_tokens=80,
                output_tokens=30,
                model="gpt-4o",
                llm_input=[{"role": "user", "content": "hi"}],
                llm_output="hello there",
                cache_read_tokens=20,
                cache_creation_tokens=0,
            )

            event = mongo_db.events.find_one({"run_id": run_id, "event_type": "node_end"})
            assert event["latency_ms"] == pytest.approx(250.5)
            assert event["input_tokens"] == 80
            assert event["output_tokens"] == 30
            assert event["model"] == "gpt-4o"
            assert event["llm_input"] == [{"role": "user", "content": "hi"}]
            assert event["llm_output"] == "hello there"
            assert event["cache_read_tokens"] == 20
        finally:
            mongo_db.runs.delete_one({"_id": run_id})
            mongo_db.events.delete_many({"run_id": run_id})

    def test_error_event_stored_with_error_field(self, mongo_db, run_id):
        try:
            mongo_db.create_run(run_id, "g", "t")
            mongo_db.insert_event(
                run_id=run_id,
                graph_id="g",
                event_type="error",
                node_name="agent",
                error="TimeoutError: request timed out",
            )

            event = mongo_db.events.find_one({"run_id": run_id, "event_type": "error"})
            assert event is not None
            assert event["error"] == "TimeoutError: request timed out"
        finally:
            mongo_db.runs.delete_one({"_id": run_id})
            mongo_db.events.delete_many({"run_id": run_id})


@pytest.mark.integration
class TestMongoFullLifecycle:
    def test_complete_run_lifecycle_event_count(self, mongo_db, run_id):
        try:
            mongo_db.create_run(run_id, "my_graph", "thread-1")
            mongo_db.insert_event(run_id=run_id, graph_id="my_graph", event_type="node_start", node_name="agent")
            mongo_db.insert_event(
                run_id=run_id,
                graph_id="my_graph",
                event_type="node_end",
                node_name="agent",
                latency_ms=100.0,
                input_tokens=50,
                output_tokens=20,
            )
            mongo_db.insert_event(run_id=run_id, graph_id="my_graph", event_type="tool_call", node_name="search")
            mongo_db.insert_event(
                run_id=run_id, graph_id="my_graph", event_type="tool_result", node_name="search", latency_ms=30.0
            )
            mongo_db.complete_run(run_id, total_input_tokens=50, total_output_tokens=20)

            run_doc = mongo_db.runs.find_one({"_id": run_id})
            assert run_doc["status"] == "completed"

            event_count = mongo_db.events.count_documents({"run_id": run_id})
            assert event_count == 4
        finally:
            mongo_db.runs.delete_one({"_id": run_id})
            mongo_db.events.delete_many({"run_id": run_id})
