from __future__ import annotations

import time
from unittest.mock import MagicMock
from uuid import UUID, uuid4

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from stakeout_agent.callback_handler import AsyncLangGraphMonitorCallback, LangGraphMonitorCallback, _MonitorBase
from stakeout_agent.callback_handler.base import _extract_cache_tokens


def make_uuid() -> UUID:
    return uuid4()


def mock_db() -> MagicMock:
    return MagicMock()


# ---------------------------------------------------------------------------
# _MonitorBase helpers
# ---------------------------------------------------------------------------


class TestExtractName:
    def test_from_serialized_name(self):
        assert _MonitorBase._extract_name({"name": "my_node"}, {}) == "my_node"

    def test_from_serialized_id(self):
        assert _MonitorBase._extract_name({"id": ["pkg", "MyNode"]}, {}) == "MyNode"

    def test_from_kwargs_when_no_serialized(self):
        assert _MonitorBase._extract_name(None, {"name": "fallback"}) == "fallback"

    def test_unknown_when_nothing_available(self):
        assert _MonitorBase._extract_name(None, {}) == "unknown"

    def test_empty_id_list_falls_back_to_unknown(self):
        # Regression: empty "id" list previously raised IndexError on [-1].
        assert _MonitorBase._extract_name({"id": []}, {}) == "unknown"


class TestPopLatency:
    def test_returns_positive_milliseconds(self):
        store = {"k": time.monotonic() - 0.1}
        latency = _MonitorBase._pop_latency(store, "k")
        assert latency is not None
        assert latency >= 100.0
        assert "k" not in store

    def test_missing_key_returns_none(self):
        assert _MonitorBase._pop_latency({}, "missing") is None


class TestSafeTruncate:
    def test_short_string_returned_unchanged(self):
        assert _MonitorBase._safe_truncate("hello") == "hello"

    def test_long_string_truncated(self):
        long_str = "x" * 600
        result = _MonitorBase._safe_truncate(long_str)
        assert len(result) == 500

    def test_dict_serialised_to_json_string(self):
        data = {"key": "value"}
        assert _MonitorBase._safe_truncate(data) == '{"key": "value"}'

    def test_non_serialisable_object_converted_via_str(self):
        class Custom:
            def __str__(self):
                return "custom-repr"

        result = _MonitorBase._safe_truncate({"obj": Custom()})
        assert isinstance(result, str)
        assert "custom-repr" in result

    def test_unserializable_str_returns_empty_string(self):
        class Bad:
            def __str__(self):
                raise RuntimeError("boom")

            def __repr__(self):
                raise RuntimeError("boom")

        assert _MonitorBase._safe_truncate(Bad()) == ""

    def test_always_returns_str(self):
        for value in ["hello", 42, {"key": "val"}, [1, 2, 3], None]:
            assert isinstance(_MonitorBase._safe_truncate(value), str)

    def test_langchain_message_like_object_serialisable(self):
        class HumanMessage:
            def __init__(self, content):
                self.content = content

            def __str__(self):
                return f"HumanMessage(content={self.content!r})"

        inputs = {"messages": [HumanMessage("hi")], "user": "alice"}
        result = _MonitorBase._safe_truncate(inputs)
        assert isinstance(result, str)
        import json

        # Result must be valid JSON (from the json.dumps path with default=str)
        parsed = json.loads(result)
        assert "messages" in parsed


class TestExtractMessages:
    def test_returns_none_for_non_dict(self):
        assert _MonitorBase._extract_messages("text") is None

    def test_returns_none_when_no_messages_key(self):
        assert _MonitorBase._extract_messages({"value": 42}) is None

    def test_returns_none_for_empty_messages(self):
        assert _MonitorBase._extract_messages({"messages": []}) is None

    def test_extracts_plain_dicts(self):
        state = {"messages": [{"role": "human", "content": "hello"}]}
        result = _MonitorBase._extract_messages(state)
        assert result == [{"role": "human", "content": "hello"}]

    def test_extracts_langchain_message_objects(self):
        class FakeHumanMessage:
            type = "human"
            content = "hi"

        class FakeAIMessage:
            type = "ai"
            content = "hello back"

        state = {"messages": [FakeHumanMessage(), FakeAIMessage()]}
        result = _MonitorBase._extract_messages(state)
        assert result == [
            {"role": "human", "content": "hi"},
            {"role": "assistant", "content": "hello back"},
        ]

    def test_truncates_long_content(self):
        state = {"messages": [{"role": "human", "content": "x" * 600}]}
        result = _MonitorBase._extract_messages(state)
        assert result is not None
        assert len(result[0]["content"]) == 500


# ---------------------------------------------------------------------------
# LangGraphMonitorCallback (sync)
# ---------------------------------------------------------------------------


GRAPH_ID = "test_graph"
THREAD_ID = "thread_1"


class TestSyncCallback:
    def _make(self) -> tuple[LangGraphMonitorCallback, MagicMock]:
        db = mock_db()
        cb = LangGraphMonitorCallback(graph_id=GRAPH_ID, thread_id=THREAD_ID, db=db)
        return cb, db

    def test_on_chain_start_root_creates_run(self):
        cb, db = self._make()
        run_id = make_uuid()
        cb.on_chain_start({}, {}, run_id=run_id, parent_run_id=None)
        db.create_run.assert_called_once_with(
            str(run_id), GRAPH_ID, THREAD_ID,
            run_inputs="{}", parent_run_id=None, prompt_version=None,
        )
        assert str(run_id) in cb._active_runs

    def test_on_chain_start_node_inserts_event(self):
        cb, db = self._make()
        root_id = make_uuid()
        node_id = make_uuid()
        cb.on_chain_start({}, {}, run_id=root_id, parent_run_id=None)
        cb.on_chain_start({"name": "my_node"}, {"x": 1}, run_id=node_id, parent_run_id=root_id)
        db.insert_event.assert_called_once()
        kwargs = db.insert_event.call_args.kwargs
        assert kwargs["event_type"] == "node_start"
        assert kwargs["node_name"] == "my_node"

    def test_on_chain_start_node_passes_messages_when_present(self):
        class FakeHumanMessage:
            type = "human"
            content = "hello"

        cb, db = self._make()
        root_id = make_uuid()
        node_id = make_uuid()
        cb.on_chain_start({}, {}, run_id=root_id, parent_run_id=None)
        cb.on_chain_start(
            {"name": "n"},
            {"messages": [FakeHumanMessage()]},
            run_id=node_id,
            parent_run_id=root_id,
        )
        kwargs = db.insert_event.call_args.kwargs
        assert kwargs["messages"] == [{"role": "human", "content": "hello"}]

    def test_on_chain_start_node_passes_no_messages_when_absent(self):
        cb, db = self._make()
        root_id = make_uuid()
        node_id = make_uuid()
        cb.on_chain_start({}, {}, run_id=root_id, parent_run_id=None)
        cb.on_chain_start({"name": "n"}, {"value": 5}, run_id=node_id, parent_run_id=root_id)
        kwargs = db.insert_event.call_args.kwargs
        assert kwargs["messages"] is None

    def test_on_chain_end_root_completes_run(self):
        cb, db = self._make()
        run_id = make_uuid()
        cb.on_chain_start({}, {}, run_id=run_id, parent_run_id=None)
        cb.on_chain_end({}, run_id=run_id, parent_run_id=None)
        db.complete_run.assert_called_once()
        assert db.complete_run.call_args.args[0] == str(run_id)

    def test_on_chain_end_node_inserts_event_with_latency(self):
        cb, db = self._make()
        root_id = make_uuid()
        node_id = make_uuid()
        cb.on_chain_start({}, {}, run_id=root_id, parent_run_id=None)
        cb.on_chain_start({"name": "n"}, {}, run_id=node_id, parent_run_id=root_id)
        cb.on_chain_end({}, run_id=node_id, parent_run_id=root_id, name="n")
        kwargs = db.insert_event.call_args_list[-1].kwargs
        assert kwargs["event_type"] == "node_end"
        assert kwargs["latency_ms"] is not None

    def test_on_chain_error_root_fails_run(self):
        cb, db = self._make()
        run_id = make_uuid()
        cb.on_chain_start({}, {}, run_id=run_id, parent_run_id=None)
        cb.on_chain_error(ValueError("bad"), run_id=run_id, parent_run_id=None)
        db.fail_run.assert_called_once()
        assert "ValueError" in db.fail_run.call_args.args[1]

    def test_on_chain_error_node_inserts_error_event(self):
        cb, db = self._make()
        root_id = make_uuid()
        node_id = make_uuid()
        cb.on_chain_start({}, {}, run_id=root_id, parent_run_id=None)
        cb.on_chain_start({"name": "n"}, {}, run_id=node_id, parent_run_id=root_id)
        cb.on_chain_error(RuntimeError("oops"), run_id=node_id, parent_run_id=root_id, name="n")
        kwargs = db.insert_event.call_args_list[-1].kwargs
        assert kwargs["event_type"] == "error"
        assert "RuntimeError" in kwargs["error"]

    def test_on_tool_start_inserts_tool_call_event(self):
        cb, db = self._make()
        root_id = make_uuid()
        tool_id = make_uuid()
        cb.on_chain_start({}, {}, run_id=root_id, parent_run_id=None)
        cb.on_tool_start({"name": "search"}, "query", run_id=tool_id, parent_run_id=root_id)
        kwargs = db.insert_event.call_args.kwargs
        assert kwargs["event_type"] == "tool_call"
        assert kwargs["node_name"] == "search"

    def test_on_tool_end_inserts_tool_result_event(self):
        cb, db = self._make()
        root_id = make_uuid()
        tool_id = make_uuid()
        cb.on_chain_start({}, {}, run_id=root_id, parent_run_id=None)
        cb.on_tool_start({"name": "search"}, "query", run_id=tool_id, parent_run_id=root_id)
        cb.on_tool_end("result", run_id=tool_id, name="search")
        kwargs = db.insert_event.call_args.kwargs
        assert kwargs["event_type"] == "tool_result"

    def test_on_tool_error_inserts_error_event(self):
        cb, db = self._make()
        root_id = make_uuid()
        tool_id = make_uuid()
        cb.on_chain_start({}, {}, run_id=root_id, parent_run_id=None)
        cb.on_tool_start({"name": "search"}, "query", run_id=tool_id, parent_run_id=root_id)
        cb.on_tool_error(OSError("network"), run_id=tool_id, name="search")
        kwargs = db.insert_event.call_args.kwargs
        assert kwargs["event_type"] == "error"
        assert "OSError" in kwargs["error"]


# ---------------------------------------------------------------------------
# AsyncLangGraphMonitorCallback
# ---------------------------------------------------------------------------


class TestAsyncCallback:
    def _make(self) -> tuple[AsyncLangGraphMonitorCallback, MagicMock]:
        db = mock_db()
        cb = AsyncLangGraphMonitorCallback(graph_id=GRAPH_ID, thread_id=THREAD_ID, db=db)
        return cb, db

    async def test_on_chain_start_root_creates_run(self):
        cb, db = self._make()
        run_id = make_uuid()
        await cb.on_chain_start({}, {}, run_id=run_id, parent_run_id=None)
        db.create_run.assert_called_once_with(
            str(run_id), GRAPH_ID, THREAD_ID,
            run_inputs="{}", parent_run_id=None, prompt_version=None,
        )

    async def test_on_chain_end_root_completes_run(self):
        cb, db = self._make()
        run_id = make_uuid()
        await cb.on_chain_start({}, {}, run_id=run_id, parent_run_id=None)
        await cb.on_chain_end({}, run_id=run_id, parent_run_id=None)
        db.complete_run.assert_called_once()
        assert db.complete_run.call_args.args[0] == str(run_id)

    async def test_on_chain_error_root_fails_run(self):
        cb, db = self._make()
        run_id = make_uuid()
        await cb.on_chain_start({}, {}, run_id=run_id, parent_run_id=None)
        await cb.on_chain_error(ValueError("bad"), run_id=run_id, parent_run_id=None)
        db.fail_run.assert_called_once()

    async def test_on_tool_start_and_end(self):
        cb, db = self._make()
        root_id = make_uuid()
        tool_id = make_uuid()
        await cb.on_chain_start({}, {}, run_id=root_id, parent_run_id=None)
        await cb.on_tool_start({"name": "calc"}, "1+1", run_id=tool_id, parent_run_id=root_id)
        await cb.on_tool_end("2", run_id=tool_id, name="calc")
        events = [c.kwargs["event_type"] for c in db.insert_event.call_args_list]
        assert "tool_call" in events
        assert "tool_result" in events


# ---------------------------------------------------------------------------
# LLM payload capture
# ---------------------------------------------------------------------------


def _make_llm_result(text: str, llm_output: dict | None = None) -> LLMResult:
    gen = ChatGeneration(message=AIMessage(content=text), text=text)
    return LLMResult(generations=[[gen]], llm_output=llm_output or {})


class FakeMessage:
    def __init__(self, role_type: str, content: str):
        self.type = role_type
        self.content = content


class TestLLMPayloadCapture:
    def _make(self, **kwargs) -> tuple[LangGraphMonitorCallback, MagicMock]:
        db = mock_db()
        cb = LangGraphMonitorCallback(graph_id=GRAPH_ID, thread_id=THREAD_ID, db=db, **kwargs)
        return cb, db

    def _run_node_with_llm(self, cb, db, messages, response_text):
        root_id = make_uuid()
        node_id = make_uuid()
        cb.on_chain_start({}, {}, run_id=root_id, parent_run_id=None)
        cb.on_chain_start({"name": "agent"}, {}, run_id=node_id, parent_run_id=root_id)
        cb.on_chat_model_start({}, [messages], run_id=make_uuid(), parent_run_id=node_id)
        cb.on_llm_end(_make_llm_result(response_text), run_id=make_uuid(), parent_run_id=node_id)
        cb.on_chain_end({}, run_id=node_id, parent_run_id=root_id)
        return db.insert_event.call_args_list[-1].kwargs

    def test_llm_input_captured_on_node_end(self):
        cb, db = self._make()
        msgs = [FakeMessage("system", "You are helpful."), FakeMessage("human", "Hi")]
        kwargs = self._run_node_with_llm(cb, db, msgs, "Hello!")
        assert kwargs["llm_input"] == [
            {"role": "system", "content": "You are helpful."},
            {"role": "human", "content": "Hi"},
        ]

    def test_llm_output_captured_on_node_end(self):
        cb, db = self._make()
        kwargs = self._run_node_with_llm(cb, db, [FakeMessage("human", "ping")], "pong")
        assert kwargs["llm_output"] == "pong"

    def test_capture_payloads_false_omits_llm_fields(self):
        cb, db = self._make(capture_payloads=False)
        kwargs = self._run_node_with_llm(cb, db, [FakeMessage("human", "hello")], "hi")
        assert kwargs["llm_input"] is None
        assert kwargs["llm_output"] is None

    def test_max_payload_chars_truncates_input_content(self):
        cb, db = self._make(max_payload_chars=5)
        long_msg = FakeMessage("human", "x" * 100)
        kwargs = self._run_node_with_llm(cb, db, [long_msg], "short")
        assert len(kwargs["llm_input"][0]["content"]) == 5

    def test_max_payload_chars_truncates_output(self):
        cb, db = self._make(max_payload_chars=3)
        kwargs = self._run_node_with_llm(cb, db, [FakeMessage("human", "q")], "long response text")
        assert kwargs["llm_output"] == "lon"

    def test_on_llm_start_plain_prompts_stored_as_user_role(self):
        cb, db = self._make()
        root_id = make_uuid()
        node_id = make_uuid()
        cb.on_chain_start({}, {}, run_id=root_id, parent_run_id=None)
        cb.on_chain_start({"name": "n"}, {}, run_id=node_id, parent_run_id=root_id)
        cb.on_llm_start({}, ["tell me a joke"], run_id=make_uuid(), parent_run_id=node_id)
        cb.on_chain_end({}, run_id=node_id, parent_run_id=root_id)
        kwargs = db.insert_event.call_args_list[-1].kwargs
        assert kwargs["llm_input"] == [{"role": "user", "content": "tell me a joke"}]

    def test_node_without_llm_call_has_no_llm_fields(self):
        cb, db = self._make()
        root_id = make_uuid()
        node_id = make_uuid()
        cb.on_chain_start({}, {}, run_id=root_id, parent_run_id=None)
        cb.on_chain_start({"name": "n"}, {}, run_id=node_id, parent_run_id=root_id)
        cb.on_chain_end({}, run_id=node_id, parent_run_id=root_id)
        kwargs = db.insert_event.call_args_list[-1].kwargs
        assert kwargs["llm_input"] is None
        assert kwargs["llm_output"] is None

    def test_dropped_events_increments_on_db_failure(self):
        db = mock_db()
        db.insert_event.side_effect = RuntimeError("db down")
        cb = LangGraphMonitorCallback(graph_id=GRAPH_ID, thread_id=THREAD_ID, db=db)
        root_id = make_uuid()
        node_id = make_uuid()
        cb.on_chain_start({}, {}, run_id=root_id, parent_run_id=None)
        cb.on_chain_start({"name": "n"}, {}, run_id=node_id, parent_run_id=root_id)
        assert cb.dropped_events == 1

    def test_dropped_events_starts_at_zero(self):
        cb, _ = self._make()
        assert cb.dropped_events == 0

    def test_dropped_events_accumulates_across_calls(self):
        db = mock_db()
        db.insert_event.side_effect = RuntimeError("db down")
        cb = LangGraphMonitorCallback(graph_id=GRAPH_ID, thread_id=THREAD_ID, db=db)
        root_id = make_uuid()
        tool_id = make_uuid()
        cb.on_chain_start({}, {}, run_id=root_id, parent_run_id=None)
        cb.on_tool_start({"name": "t"}, "input", run_id=tool_id, parent_run_id=root_id)
        cb.on_tool_end("result", run_id=tool_id, name="t")
        assert cb.dropped_events == 2

    def test_tags_included_in_node_start_payload(self):
        cb, db = self._make()
        root_id = make_uuid()
        node_id = make_uuid()
        cb.on_chain_start({}, {}, run_id=root_id, parent_run_id=None)
        cb.on_chain_start({"name": "n"}, {}, run_id=node_id, parent_run_id=root_id, tags=["prod", "v2"])
        kwargs = db.insert_event.call_args.kwargs
        assert kwargs["payload"]["tags"] == ["prod", "v2"]

    def test_tags_absent_when_not_provided(self):
        cb, db = self._make()
        root_id = make_uuid()
        node_id = make_uuid()
        cb.on_chain_start({}, {}, run_id=root_id, parent_run_id=None)
        cb.on_chain_start({"name": "n"}, {}, run_id=node_id, parent_run_id=root_id)
        kwargs = db.insert_event.call_args.kwargs
        assert "tags" not in kwargs["payload"]

    def test_on_tool_start_uses_structured_inputs_when_available(self):
        cb, db = self._make()
        root_id = make_uuid()
        tool_id = make_uuid()
        cb.on_chain_start({}, {}, run_id=root_id, parent_run_id=None)
        cb.on_tool_start({"name": "search"}, '{"q": "hello"}', run_id=tool_id, parent_run_id=root_id, inputs={"q": "hello"})
        kwargs = db.insert_event.call_args.kwargs
        assert '"q"' in kwargs["payload"]["input"]
        assert "hello" in kwargs["payload"]["input"]

    def test_on_tool_start_falls_back_to_input_str(self):
        cb, db = self._make()
        root_id = make_uuid()
        tool_id = make_uuid()
        cb.on_chain_start({}, {}, run_id=root_id, parent_run_id=None)
        cb.on_tool_start({"name": "search"}, "raw query", run_id=tool_id, parent_run_id=root_id)
        kwargs = db.insert_event.call_args.kwargs
        assert kwargs["payload"]["input"] == "raw query"

    def test_on_retriever_start_inserts_retriever_start_event(self):
        cb, db = self._make()
        root_id = make_uuid()
        ret_id = make_uuid()
        cb.on_chain_start({}, {}, run_id=root_id, parent_run_id=None)
        cb.on_retriever_start({"id": ["pkg", "VectorStoreRetriever"]}, "what is RAG?", run_id=ret_id, parent_run_id=root_id)
        kwargs = db.insert_event.call_args.kwargs
        assert kwargs["event_type"] == "retriever_start"
        assert kwargs["node_name"] == "VectorStoreRetriever"
        assert kwargs["payload"]["query"] == "what is RAG?"

    def test_on_retriever_end_inserts_retriever_end_event_with_latency_and_doc_count(self):
        cb, db = self._make()
        root_id = make_uuid()
        ret_id = make_uuid()
        cb.on_chain_start({}, {}, run_id=root_id, parent_run_id=None)
        cb.on_retriever_start({"id": ["pkg", "VectorStoreRetriever"]}, "query", run_id=ret_id, parent_run_id=root_id)
        cb.on_retriever_end(["doc1", "doc2", "doc3"], run_id=ret_id)
        kwargs = db.insert_event.call_args.kwargs
        assert kwargs["event_type"] == "retriever_end"
        assert kwargs["payload"]["document_count"] == 3
        assert kwargs["latency_ms"] is not None

    def test_on_retriever_error_inserts_error_event(self):
        cb, db = self._make()
        root_id = make_uuid()
        ret_id = make_uuid()
        cb.on_chain_start({}, {}, run_id=root_id, parent_run_id=None)
        cb.on_retriever_start({"id": ["pkg", "VectorStoreRetriever"]}, "query", run_id=ret_id, parent_run_id=root_id)
        cb.on_retriever_error(ConnectionError("index unavailable"), run_id=ret_id)
        kwargs = db.insert_event.call_args.kwargs
        assert kwargs["event_type"] == "error"
        assert "ConnectionError" in kwargs["error"]
        assert kwargs["latency_ms"] is not None

    def test_anthropic_cache_tokens_stored_on_node_end(self):
        cb, db = self._make()
        root_id = make_uuid()
        node_id = make_uuid()
        cb.on_chain_start({}, {}, run_id=root_id, parent_run_id=None)
        cb.on_chain_start({"name": "agent"}, {}, run_id=node_id, parent_run_id=root_id)
        llm_output = {
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_read_input_tokens": 80,
                "cache_creation_input_tokens": 20,
            }
        }
        cb.on_llm_end(_make_llm_result("ok", llm_output), run_id=make_uuid(), parent_run_id=node_id)
        cb.on_chain_end({}, run_id=node_id, parent_run_id=root_id)
        kwargs = db.insert_event.call_args_list[-1].kwargs
        assert kwargs["cache_read_tokens"] == 80
        assert kwargs["cache_creation_tokens"] == 20

    def test_openai_cache_tokens_stored_on_node_end(self):
        cb, db = self._make()
        root_id = make_uuid()
        node_id = make_uuid()
        cb.on_chain_start({}, {}, run_id=root_id, parent_run_id=None)
        cb.on_chain_start({"name": "agent"}, {}, run_id=node_id, parent_run_id=root_id)
        llm_output = {
            "token_usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "prompt_tokens_details": {"cached_tokens": 60},
            },
            "model_name": "gpt-4o",
        }
        cb.on_llm_end(_make_llm_result("ok", llm_output), run_id=make_uuid(), parent_run_id=node_id)
        cb.on_chain_end({}, run_id=node_id, parent_run_id=root_id)
        kwargs = db.insert_event.call_args_list[-1].kwargs
        assert kwargs["cache_read_tokens"] == 60
        assert kwargs["cache_creation_tokens"] is None

    def test_cache_tokens_rolled_up_to_run(self):
        cb, db = self._make()
        root_id = make_uuid()
        node_id = make_uuid()
        cb.on_chain_start({}, {}, run_id=root_id, parent_run_id=None)
        cb.on_chain_start({"name": "agent"}, {}, run_id=node_id, parent_run_id=root_id)
        llm_output = {
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_read_input_tokens": 80,
                "cache_creation_input_tokens": 20,
            }
        }
        cb.on_llm_end(_make_llm_result("ok", llm_output), run_id=make_uuid(), parent_run_id=node_id)
        cb.on_chain_end({}, run_id=node_id, parent_run_id=root_id)
        cb.on_chain_end({}, run_id=root_id, parent_run_id=None)
        kwargs = db.complete_run.call_args.kwargs
        assert kwargs["total_cache_read_tokens"] == 80
        assert kwargs["total_cache_creation_tokens"] == 20

    def test_no_cache_tokens_when_absent(self):
        cb, db = self._make()
        root_id = make_uuid()
        node_id = make_uuid()
        cb.on_chain_start({}, {}, run_id=root_id, parent_run_id=None)
        cb.on_chain_start({"name": "agent"}, {}, run_id=node_id, parent_run_id=root_id)
        llm_output = {"token_usage": {"prompt_tokens": 100, "completion_tokens": 50}, "model_name": "gpt-4o"}
        cb.on_llm_end(_make_llm_result("ok", llm_output), run_id=make_uuid(), parent_run_id=node_id)
        cb.on_chain_end({}, run_id=node_id, parent_run_id=root_id)
        kwargs = db.insert_event.call_args_list[-1].kwargs
        assert kwargs["cache_read_tokens"] is None
        assert kwargs["cache_creation_tokens"] is None


class TestExtractCacheTokens:
    def test_anthropic_both_fields(self):
        meta = {
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_read_input_tokens": 80,
                "cache_creation_input_tokens": 20,
            }
        }
        assert _extract_cache_tokens(meta) == (80, 20)

    def test_anthropic_read_only(self):
        meta = {"usage": {"input_tokens": 100, "output_tokens": 50, "cache_read_input_tokens": 40}}
        cr, cc = _extract_cache_tokens(meta)
        assert cr == 40
        assert cc is None

    def test_openai_cached_tokens(self):
        meta = {
            "token_usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "prompt_tokens_details": {"cached_tokens": 60},
            }
        }
        assert _extract_cache_tokens(meta) == (60, None)

    def test_no_cache_fields(self):
        meta = {"token_usage": {"prompt_tokens": 100, "completion_tokens": 50}}
        assert _extract_cache_tokens(meta) == (None, None)

    def test_empty_metadata(self):
        assert _extract_cache_tokens({}) == (None, None)


class TestLLMPayloadClearance:
    def _make(self, **kwargs) -> tuple[LangGraphMonitorCallback, MagicMock]:
        db = mock_db()
        return LangGraphMonitorCallback(graph_id=GRAPH_ID, thread_id=THREAD_ID, db=db, **kwargs), db

    def test_llm_inputs_cleared_after_run_completes(self):
        cb, db = self._make()
        root_id = make_uuid()
        node_id = make_uuid()
        cb.on_chain_start({}, {}, run_id=root_id, parent_run_id=None)
        cb.on_chain_start({"name": "n"}, {}, run_id=node_id, parent_run_id=root_id)
        cb.on_chat_model_start({}, [[FakeMessage("human", "hi")]], run_id=make_uuid(), parent_run_id=node_id)
        cb.on_chain_end({}, run_id=node_id, parent_run_id=root_id)
        cb.on_chain_end({}, run_id=root_id, parent_run_id=None)
        assert cb._active_runs == {}
