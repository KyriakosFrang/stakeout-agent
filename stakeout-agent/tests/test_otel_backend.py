from __future__ import annotations

from unittest.mock import MagicMock, patch

from stakeout_agent.backends.otel import OTELMonitorDB

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_provider():
    """Return a mock TracerProvider + Tracer + Span triple."""
    mock_span = MagicMock()
    mock_tracer = MagicMock()
    mock_tracer.start_span.return_value = mock_span
    mock_provider = MagicMock()
    mock_provider.get_tracer.return_value = mock_tracer
    return mock_provider, mock_tracer, mock_span


def _db(provider=None):
    if provider is None:
        provider, _, _ = _make_mock_provider()
    return OTELMonitorDB(tracer_provider=provider)


# ---------------------------------------------------------------------------
# create_run
# ---------------------------------------------------------------------------


class TestCreateRun:
    def test_starts_root_span_with_graph_id_as_name(self):
        provider, tracer, _ = _make_mock_provider()
        db = _db(provider)
        db.create_run("run-1", "my_graph", "thread-42")

        tracer.start_span.assert_called_once()
        assert tracer.start_span.call_args.args[0] == "my_graph"

    def test_root_span_attributes(self):
        provider, tracer, span = _make_mock_provider()
        db = _db(provider)
        db.create_run("run-1", "my_graph", "thread-42")

        call_kwargs = tracer.start_span.call_args.kwargs
        attrs = call_kwargs.get("attributes") or {}
        assert attrs["stakeout.run_id"] == "run-1"
        assert attrs["stakeout.graph_id"] == "my_graph"
        assert attrs["stakeout.thread_id"] == "thread-42"

    def test_run_span_stored_internally(self):
        provider, _, span = _make_mock_provider()
        db = _db(provider)
        db.create_run("run-1", "g", "t")
        assert db._run_spans["run-1"][0] is span


# ---------------------------------------------------------------------------
# complete_run
# ---------------------------------------------------------------------------


class TestCompleteRun:
    def test_ends_span_with_ok_status(self):
        from opentelemetry.trace import StatusCode

        provider, _, span = _make_mock_provider()
        db = _db(provider)
        db.create_run("run-1", "g", "t")
        db.complete_run("run-1")

        span.set_status.assert_called_once()
        assert span.set_status.call_args.args[0] == StatusCode.OK
        span.end.assert_called_once()

    def test_sets_token_attributes(self):
        provider, _, span = _make_mock_provider()
        db = _db(provider)
        db.create_run("run-1", "g", "t")
        db.complete_run("run-1", total_input_tokens=100, total_output_tokens=50, estimated_cost_usd=0.002)

        set_attr_calls = {c.args[0]: c.args[1] for c in span.set_attribute.call_args_list}
        assert set_attr_calls["gen_ai.usage.input_tokens"] == 100
        assert set_attr_calls["gen_ai.usage.output_tokens"] == 50
        assert set_attr_calls["stakeout.cost_usd"] == 0.002

    def test_sets_cache_token_attributes(self):
        provider, _, span = _make_mock_provider()
        db = _db(provider)
        db.create_run("run-1", "g", "t")
        db.complete_run("run-1", total_cache_read_tokens=20, total_cache_creation_tokens=5)

        set_attr_calls = {c.args[0]: c.args[1] for c in span.set_attribute.call_args_list}
        assert set_attr_calls["gen_ai.usage.cache_read_input_tokens"] == 20
        assert set_attr_calls["gen_ai.usage.cache_creation_input_tokens"] == 5

    def test_removes_span_from_internal_dict(self):
        provider, _, _ = _make_mock_provider()
        db = _db(provider)
        db.create_run("run-1", "g", "t")
        db.complete_run("run-1")
        assert "run-1" not in db._run_spans

    def test_unknown_run_id_logs_warning_does_not_raise(self, caplog):
        db = _db()
        db.complete_run("nonexistent")  # must not raise
        assert any("nonexistent" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# fail_run
# ---------------------------------------------------------------------------


class TestFailRun:
    def test_sets_error_status_and_records_exception(self):
        from opentelemetry.trace import StatusCode

        provider, _, span = _make_mock_provider()
        db = _db(provider)
        db.create_run("run-1", "g", "t")
        db.fail_run("run-1", "something went wrong")

        span.set_status.assert_called_once()
        status_code = span.set_status.call_args.args[0]
        assert status_code == StatusCode.ERROR
        span.record_exception.assert_called_once()
        exc = span.record_exception.call_args.args[0]
        assert "something went wrong" in str(exc)
        span.end.assert_called_once()

    def test_removes_span_from_internal_dict(self):
        provider, _, _ = _make_mock_provider()
        db = _db(provider)
        db.create_run("run-1", "g", "t")
        db.fail_run("run-1", "boom")
        assert "run-1" not in db._run_spans

    def test_unknown_run_id_logs_warning_does_not_raise(self, caplog):
        db = _db()
        db.fail_run("nonexistent", "error")  # must not raise
        assert any("nonexistent" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# insert_event — node lifecycle
# ---------------------------------------------------------------------------


class TestNodeSpanLifecycle:
    def test_node_start_creates_child_span(self):
        provider, tracer, root_span = _make_mock_provider()
        db = _db(provider)
        db.create_run("run-1", "g", "t")

        node_span = MagicMock()
        tracer.start_span.return_value = node_span
        db.insert_event(run_id="run-1", graph_id="g", event_type="node_start", node_name="my_node")

        assert tracer.start_span.call_count == 2  # root + node
        assert tracer.start_span.call_args.args[0] == "my_node"
        assert ("run-1", "my_node") in db._node_spans

    def test_node_end_sets_attributes_and_ends_span(self):
        provider, tracer, root_span = _make_mock_provider()
        db = _db(provider)
        db.create_run("run-1", "g", "t")

        node_span = MagicMock()
        tracer.start_span.return_value = node_span
        db.insert_event(run_id="run-1", graph_id="g", event_type="node_start", node_name="my_node")
        db.insert_event(
            run_id="run-1",
            graph_id="g",
            event_type="node_end",
            node_name="my_node",
            latency_ms=42.5,
            input_tokens=10,
            output_tokens=5,
            model="gpt-4o",
        )

        attrs = {c.args[0]: c.args[1] for c in node_span.set_attribute.call_args_list}
        assert attrs["stakeout.latency_ms"] == 42.5
        assert attrs["gen_ai.usage.input_tokens"] == 10
        assert attrs["gen_ai.usage.output_tokens"] == 5
        assert attrs["gen_ai.request.model"] == "gpt-4o"
        node_span.end.assert_called_once()
        assert ("run-1", "my_node") not in db._node_spans

    def test_node_end_with_llm_io_adds_span_events_not_attributes(self):
        provider, tracer, root_span = _make_mock_provider()
        db = _db(provider)
        db.create_run("run-1", "g", "t")

        node_span = MagicMock()
        tracer.start_span.return_value = node_span
        db.insert_event(run_id="run-1", graph_id="g", event_type="node_start", node_name="n")
        db.insert_event(
            run_id="run-1",
            graph_id="g",
            event_type="node_end",
            node_name="n",
            llm_input=[{"role": "user", "content": "hello"}],
            llm_output="world",
        )

        add_event_names = [c.args[0] for c in node_span.add_event.call_args_list]
        assert "gen_ai.content.prompt" in add_event_names
        assert "gen_ai.content.completion" in add_event_names

        # payload must not appear as a span attribute
        set_attr_names = [c.args[0] for c in node_span.set_attribute.call_args_list]
        assert "llm_input" not in set_attr_names
        assert "llm_output" not in set_attr_names

    def test_node_end_unknown_span_logs_warning(self, caplog):
        db = _db()
        db.insert_event(run_id="run-1", graph_id="g", event_type="node_end", node_name="ghost")
        assert any("ghost" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# insert_event — tool lifecycle
# ---------------------------------------------------------------------------


class TestToolSpanLifecycle:
    def test_tool_call_creates_span(self):
        provider, tracer, root_span = _make_mock_provider()
        db = _db(provider)
        db.create_run("run-1", "g", "t")

        tool_span = MagicMock()
        tracer.start_span.return_value = tool_span
        db.insert_event(run_id="run-1", graph_id="g", event_type="tool_call", node_name="search_tool")

        assert tracer.start_span.call_args.args[0] == "search_tool"
        assert ("run-1", "search_tool") in db._tool_spans

    def test_tool_result_ends_span(self):
        provider, tracer, root_span = _make_mock_provider()
        db = _db(provider)
        db.create_run("run-1", "g", "t")

        tool_span = MagicMock()
        tracer.start_span.return_value = tool_span
        db.insert_event(run_id="run-1", graph_id="g", event_type="tool_call", node_name="search_tool")
        db.insert_event(
            run_id="run-1", graph_id="g", event_type="tool_result", node_name="search_tool", latency_ms=15.0
        )

        attrs = {c.args[0]: c.args[1] for c in tool_span.set_attribute.call_args_list}
        assert attrs["stakeout.latency_ms"] == 15.0
        tool_span.end.assert_called_once()
        assert ("run-1", "search_tool") not in db._tool_spans

    def test_tool_result_unknown_span_logs_warning(self, caplog):
        db = _db()
        db.insert_event(run_id="run-1", graph_id="g", event_type="tool_result", node_name="ghost_tool")
        assert any("ghost_tool" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# insert_event — retriever lifecycle (same pattern as tool)
# ---------------------------------------------------------------------------


class TestRetrieverSpanLifecycle:
    def test_retriever_start_end(self):
        provider, tracer, root_span = _make_mock_provider()
        db = _db(provider)
        db.create_run("run-1", "g", "t")

        ret_span = MagicMock()
        tracer.start_span.return_value = ret_span
        db.insert_event(run_id="run-1", graph_id="g", event_type="retriever_start", node_name="vector_store")
        db.insert_event(
            run_id="run-1", graph_id="g", event_type="retriever_end", node_name="vector_store", latency_ms=8.0
        )

        ret_span.end.assert_called_once()
        assert ("run-1", "vector_store") not in db._tool_spans


# ---------------------------------------------------------------------------
# insert_event — error
# ---------------------------------------------------------------------------


class TestErrorEvent:
    def test_error_on_node_span_sets_error_status(self):
        from opentelemetry.trace import StatusCode

        provider, tracer, root_span = _make_mock_provider()
        db = _db(provider)
        db.create_run("run-1", "g", "t")

        node_span = MagicMock()
        tracer.start_span.return_value = node_span
        db.insert_event(run_id="run-1", graph_id="g", event_type="node_start", node_name="fail_node")
        db.insert_event(
            run_id="run-1", graph_id="g", event_type="error", node_name="fail_node", error="ValueError: oops"
        )

        status_call = node_span.set_status.call_args.args[0]
        assert status_call == StatusCode.ERROR
        node_span.record_exception.assert_called_once()
        node_span.end.assert_called_once()
        assert ("run-1", "fail_node") not in db._node_spans

    def test_error_on_tool_span_sets_error_status(self):
        from opentelemetry.trace import StatusCode

        provider, tracer, root_span = _make_mock_provider()
        db = _db(provider)
        db.create_run("run-1", "g", "t")

        tool_span = MagicMock()
        tracer.start_span.return_value = tool_span
        db.insert_event(run_id="run-1", graph_id="g", event_type="tool_call", node_name="bad_tool")
        db.insert_event(run_id="run-1", graph_id="g", event_type="error", node_name="bad_tool", error="TimeoutError")

        status_call = tool_span.set_status.call_args.args[0]
        assert status_call == StatusCode.ERROR
        tool_span.end.assert_called_once()
        assert ("run-1", "bad_tool") not in db._tool_spans

    def test_error_unknown_span_logs_warning(self, caplog):
        db = _db()
        db.insert_event(run_id="run-1", graph_id="g", event_type="error", node_name="ghost", error="boom")
        assert any("ghost" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Provider resolution
# ---------------------------------------------------------------------------


class TestProviderResolution:
    def test_explicit_provider_is_used(self):
        provider, tracer, _ = _make_mock_provider()
        db = OTELMonitorDB(tracer_provider=provider)
        assert db._provider is provider

    def test_auto_configure_when_otlp_endpoint_set(self, monkeypatch):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        mock_provider = MagicMock()
        with patch("stakeout_agent.backends.otel._build_default_provider", return_value=mock_provider) as mock_build:
            db = OTELMonitorDB()
        mock_build.assert_called_once_with("stakeout-agent")
        assert db._provider is mock_provider

    def test_falls_back_to_global_provider_when_no_endpoint(self, monkeypatch):
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        mock_global = MagicMock()
        with patch("opentelemetry.trace.get_tracer_provider", return_value=mock_global):
            db = OTELMonitorDB()
        assert db._provider is mock_global

    def test_service_name_from_env(self, monkeypatch):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        monkeypatch.setenv("OTEL_SERVICE_NAME", "my-custom-service")
        with patch("stakeout_agent.backends.otel._build_default_provider", return_value=MagicMock()) as mock_build:
            OTELMonitorDB()
        mock_build.assert_called_once_with("my-custom-service")

    def test_explicit_service_name_overrides_env(self, monkeypatch):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        monkeypatch.setenv("OTEL_SERVICE_NAME", "env-service")
        with patch("stakeout_agent.backends.otel._build_default_provider", return_value=MagicMock()) as mock_build:
            OTELMonitorDB(service_name="explicit-service")
        mock_build.assert_called_once_with("explicit-service")


# ---------------------------------------------------------------------------
# Unknown event_type is silently ignored
# ---------------------------------------------------------------------------


class TestUnknownEventType:
    def test_unknown_event_type_does_not_raise(self):
        db = _db()
        db.insert_event(run_id="r", graph_id="g", event_type="custom_event", node_name="n")


# ---------------------------------------------------------------------------
# Stale span reaping
# ---------------------------------------------------------------------------


class TestStaleSpanReaping:
    def test_stale_run_span_is_ended_with_error_on_next_create_run(self):
        from opentelemetry.trace import StatusCode

        provider, tracer, stale_span = _make_mock_provider()
        db = OTELMonitorDB(tracer_provider=provider, stale_span_ttl_seconds=10.0)
        db.create_run("stale-run", "g", "t")

        run_id, (span, start) = next(iter(db._run_spans.items()))
        db._run_spans[run_id] = (span, start - 100.0)

        new_span = MagicMock()
        tracer.start_span.return_value = new_span
        db.create_run("new-run", "g", "t")

        stale_span.set_status.assert_called_once()
        assert stale_span.set_status.call_args.args[0] == StatusCode.ERROR
        stale_span.end.assert_called_once()
        assert "stale-run" not in db._run_spans
        assert "new-run" in db._run_spans

    def test_stale_node_and_tool_spans_are_ended(self):
        provider, tracer, _ = _make_mock_provider()
        db = OTELMonitorDB(tracer_provider=provider, stale_span_ttl_seconds=10.0)
        db.create_run("run-1", "g", "t")

        node_span = MagicMock()
        tracer.start_span.return_value = node_span
        db.insert_event(run_id="run-1", graph_id="g", event_type="node_start", node_name="my_node")

        tool_span = MagicMock()
        tracer.start_span.return_value = tool_span
        db.insert_event(run_id="run-1", graph_id="g", event_type="tool_call", node_name="my_tool")

        # backdate both child spans past the TTL
        key = ("run-1", "my_node")
        span, start = db._node_spans[key]
        db._node_spans[key] = (span, start - 100.0)
        key = ("run-1", "my_tool")
        span, start = db._tool_spans[key]
        db._tool_spans[key] = (span, start - 100.0)

        db.create_run("run-2", "g", "t")

        node_span.end.assert_called_once()
        tool_span.end.assert_called_once()
        assert ("run-1", "my_node") not in db._node_spans
        assert ("run-1", "my_tool") not in db._tool_spans

    def test_span_within_ttl_is_not_reaped(self):
        provider, _, span = _make_mock_provider()
        db = OTELMonitorDB(tracer_provider=provider, stale_span_ttl_seconds=3600.0)
        db.create_run("run-1", "g", "t")

        db.create_run("run-2", "g", "t")

        span.end.assert_not_called()
        assert "run-1" in db._run_spans

    def test_ttl_none_disables_reaping(self):
        provider, _, span = _make_mock_provider()
        db = OTELMonitorDB(tracer_provider=provider, stale_span_ttl_seconds=None)
        db.create_run("run-1", "g", "t")

        run_id, (s, start) = next(iter(db._run_spans.items()))
        db._run_spans[run_id] = (s, start - 10_000.0)

        db.create_run("run-2", "g", "t")

        span.end.assert_not_called()
        assert "run-1" in db._run_spans
