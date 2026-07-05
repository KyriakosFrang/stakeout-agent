from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import TYPE_CHECKING, Any

from stakeout_agent.backends.base import AbstractMonitorDB

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.trace import Span

_logger = logging.getLogger(__name__)


def _build_default_provider(service_name: str) -> Any:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider as SdkTracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resource = Resource.create({"service.name": service_name})
    provider = SdkTracerProvider(resource=resource)
    exporter = OTLPSpanExporter()
    provider.add_span_processor(BatchSpanProcessor(exporter))
    return provider


class OTELMonitorDB(AbstractMonitorDB):
    """OpenTelemetry backend — maps stakeout runs/events to OTEL traces/spans.

    Drop-in replacement for MongoDB/Postgres backends that exports to any
    OTEL-compatible collector (Jaeger, Honeycomb, Datadog, Grafana Tempo, …).

    Provider resolution order:
    1. Explicit ``tracer_provider`` argument
    2. Auto-configure using OTLPSpanExporter when ``OTEL_EXPORTER_OTLP_ENDPOINT`` is set
    3. Global OTEL tracer provider (``trace.get_tracer_provider()``)
    """

    INSTRUMENTATION_NAME = "stakeout-agent"

    def __init__(
        self,
        tracer_provider: TracerProvider | None = None,
        service_name: str | None = None,
        stale_span_ttl_seconds: float | None = 3600.0,
    ) -> None:
        from opentelemetry import trace

        self._service_name = service_name or os.environ.get("OTEL_SERVICE_NAME", "stakeout-agent")

        if tracer_provider is not None:
            self._provider = tracer_provider
        elif os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
            self._provider = _build_default_provider(self._service_name)
        else:
            self._provider = trace.get_tracer_provider()

        self._tracer = self._provider.get_tracer(self.INSTRUMENTATION_NAME)

        self._stale_span_ttl_seconds = stale_span_ttl_seconds
        self._lock = threading.Lock()
        self._run_spans: dict[str, tuple[Span, float]] = {}
        self._node_spans: dict[tuple[str, str], tuple[Span, float]] = {}
        self._tool_spans: dict[tuple[str, str], tuple[Span, float]] = {}

    def _reap_stale_spans(self, now: float) -> None:
        """End and drop spans that have been open longer than the TTL.

        Called from the top of create_run, so this leak cannot grow unbounded
        between legitimate root-run invocations.
        """
        from opentelemetry.trace import StatusCode

        if self._stale_span_ttl_seconds is None:
            return
        with self._lock:
            for store, kind in (
                (self._run_spans, "run"),
                (self._node_spans, "node"),
                (self._tool_spans, "tool"),
            ):
                stale_keys = [k for k, (_, start) in store.items() if now - start > self._stale_span_ttl_seconds]
                for k in stale_keys:
                    span, _ = store.pop(k)
                    span.set_status(StatusCode.ERROR, description="StaleSpanTimeout")
                    span.end()
                    _logger.warning("otel: reaped stale %s span key=%s", kind, k)

    # ------------------------------------------------------------------
    # AbstractMonitorDB interface
    # ------------------------------------------------------------------

    def create_run(
        self,
        run_id: str,
        graph_id: str,
        thread_id: str,
        run_inputs: str | None = None,
        parent_run_id: str | None = None,
        prompt_version: str | None = None,
        environment: str | None = None,
    ) -> None:
        self._reap_stale_spans(time.monotonic())
        attrs: dict[str, str] = {
            "stakeout.run_id": run_id,
            "stakeout.graph_id": graph_id,
            "stakeout.thread_id": thread_id,
        }
        if run_inputs is not None:
            attrs["stakeout.run_inputs"] = run_inputs
        if parent_run_id is not None:
            attrs["stakeout.parent_run_id"] = parent_run_id
        if prompt_version is not None:
            attrs["stakeout.prompt_version"] = prompt_version
        span = self._tracer.start_span(graph_id, attributes=attrs)
        with self._lock:
            self._run_spans[run_id] = (span, time.monotonic())
        _logger.debug("otel: root span started run_id=%s graph_id=%s", run_id, graph_id)

    def complete_run(
        self,
        run_id: str,
        total_input_tokens: int | None = None,
        total_output_tokens: int | None = None,
        estimated_cost_usd: float | None = None,
        total_cache_read_tokens: int | None = None,
        total_cache_creation_tokens: int | None = None,
    ) -> None:
        from opentelemetry.trace import StatusCode

        with self._lock:
            entry = self._run_spans.pop(run_id, None)
        span = entry[0] if entry is not None else None

        if span is None:
            _logger.warning("otel: complete_run called for unknown run_id=%s", run_id)
            return

        if total_input_tokens is not None:
            span.set_attribute("gen_ai.usage.input_tokens", total_input_tokens)
        if total_output_tokens is not None:
            span.set_attribute("gen_ai.usage.output_tokens", total_output_tokens)
        if total_cache_read_tokens is not None:
            span.set_attribute("gen_ai.usage.cache_read_input_tokens", total_cache_read_tokens)
        if total_cache_creation_tokens is not None:
            span.set_attribute("gen_ai.usage.cache_creation_input_tokens", total_cache_creation_tokens)
        if estimated_cost_usd is not None:
            span.set_attribute("stakeout.cost_usd", estimated_cost_usd)

        span.set_status(StatusCode.OK)
        span.end()
        _logger.debug("otel: root span ended run_id=%s", run_id)

    def prune_runs(self, older_than_days: int) -> int:
        return 0

    def fail_run(self, run_id: str, error: str) -> None:
        from opentelemetry.trace import StatusCode

        with self._lock:
            entry = self._run_spans.pop(run_id, None)
        span = entry[0] if entry is not None else None

        if span is None:
            _logger.warning("otel: fail_run called for unknown run_id=%s", run_id)
            return

        span.set_status(StatusCode.ERROR, description=error)
        span.record_exception(RuntimeError(error))
        span.end()
        _logger.debug("otel: root span failed run_id=%s error=%s", run_id, error)

    def insert_event(
        self,
        run_id: str,
        graph_id: str,
        event_type: str,
        node_name: str,
        latency_ms: float | None = None,
        payload: dict | None = None,
        error: str | None = None,
        messages: list[dict] | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        model: str | None = None,
        llm_input: list[dict] | None = None,
        llm_output: str | None = None,
        cache_read_tokens: int | None = None,
        cache_creation_tokens: int | None = None,
    ) -> None:
        handler = {
            "node_start": self._on_node_start,
            "node_end": self._on_node_end,
            "tool_call": self._on_tool_start,
            "tool_result": self._on_tool_end,
            "retriever_start": self._on_tool_start,
            "retriever_end": self._on_tool_end,
            "error": self._on_error,
        }.get(event_type)

        if handler is None:
            _logger.debug("otel: unhandled event_type=%s node=%s run_id=%s", event_type, node_name, run_id)
            return

        handler(
            run_id=run_id,
            graph_id=graph_id,
            event_type=event_type,
            node_name=node_name,
            latency_ms=latency_ms,
            payload=payload,
            error=error,
            messages=messages,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
            llm_input=llm_input,
            llm_output=llm_output,
            cache_read_tokens=cache_read_tokens,
            cache_creation_tokens=cache_creation_tokens,
        )

    # ------------------------------------------------------------------
    # Internal event handlers
    # ------------------------------------------------------------------

    def _on_node_start(self, *, run_id: str, node_name: str, **_: Any) -> None:
        from opentelemetry import trace

        with self._lock:
            entry = self._run_spans.get(run_id)
        root = entry[0] if entry is not None else None

        if root is None:
            _logger.warning("otel: node_start for unknown run_id=%s node=%s", run_id, node_name)
            return

        ctx = trace.set_span_in_context(root)
        span = self._tracer.start_span(node_name, context=ctx)
        span.set_attribute("stakeout.event_type", "node_start")
        span.set_attribute("stakeout.node_name", node_name)

        with self._lock:
            self._node_spans[(run_id, node_name)] = (span, time.monotonic())

    def _on_node_end(
        self,
        *,
        run_id: str,
        node_name: str,
        latency_ms: float | None,
        input_tokens: int | None,
        output_tokens: int | None,
        model: str | None,
        llm_input: list[dict] | None,
        llm_output: str | None,
        cache_read_tokens: int | None,
        cache_creation_tokens: int | None,
        **_: Any,
    ) -> None:
        from opentelemetry.trace import StatusCode

        with self._lock:
            entry = self._node_spans.pop((run_id, node_name), None)
        span = entry[0] if entry is not None else None

        if span is None:
            _logger.warning("otel: node_end for unknown span run_id=%s node=%s", run_id, node_name)
            return

        span.set_attribute("stakeout.event_type", "node_end")
        if latency_ms is not None:
            span.set_attribute("stakeout.latency_ms", latency_ms)
        if model is not None:
            span.set_attribute("gen_ai.request.model", model)
        if input_tokens is not None:
            span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
        if output_tokens is not None:
            span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
        if cache_read_tokens is not None:
            span.set_attribute("gen_ai.usage.cache_read_input_tokens", cache_read_tokens)
        if cache_creation_tokens is not None:
            span.set_attribute("gen_ai.usage.cache_creation_input_tokens", cache_creation_tokens)

        if llm_input is not None:
            span.add_event("gen_ai.content.prompt", {"body": json.dumps(llm_input, default=str)})
        if llm_output is not None:
            span.add_event("gen_ai.content.completion", {"body": llm_output})

        span.set_status(StatusCode.OK)
        span.end()

    def _on_tool_start(self, *, run_id: str, node_name: str, event_type: str, **_: Any) -> None:
        from opentelemetry import trace

        with self._lock:
            entry = self._run_spans.get(run_id)
        root = entry[0] if entry is not None else None

        if root is None:
            _logger.warning("otel: %s for unknown run_id=%s tool=%s", event_type, run_id, node_name)
            return

        ctx = trace.set_span_in_context(root)
        span = self._tracer.start_span(node_name, context=ctx)
        span.set_attribute("stakeout.event_type", event_type)
        span.set_attribute("stakeout.node_name", node_name)

        with self._lock:
            self._tool_spans[(run_id, node_name)] = (span, time.monotonic())

    def _on_tool_end(
        self,
        *,
        run_id: str,
        node_name: str,
        event_type: str,
        latency_ms: float | None,
        **_: Any,
    ) -> None:
        from opentelemetry.trace import StatusCode

        with self._lock:
            entry = self._tool_spans.pop((run_id, node_name), None)
        span = entry[0] if entry is not None else None

        if span is None:
            _logger.warning("otel: %s for unknown span run_id=%s tool=%s", event_type, run_id, node_name)
            return

        span.set_attribute("stakeout.event_type", event_type)
        if latency_ms is not None:
            span.set_attribute("stakeout.latency_ms", latency_ms)

        span.set_status(StatusCode.OK)
        span.end()

    def _on_error(self, *, run_id: str, node_name: str, error: str | None, latency_ms: float | None, **_: Any) -> None:
        from opentelemetry.trace import StatusCode

        with self._lock:
            entry = self._node_spans.pop((run_id, node_name), None) or self._tool_spans.pop((run_id, node_name), None)
        span = entry[0] if entry is not None else None

        if span is None:
            _logger.warning("otel: error event for unknown span run_id=%s node=%s", run_id, node_name)
            return

        error_msg = error or "unknown error"
        span.set_attribute("stakeout.event_type", "error")
        if latency_ms is not None:
            span.set_attribute("stakeout.latency_ms", latency_ms)
        span.set_status(StatusCode.ERROR, description=error_msg)
        span.record_exception(RuntimeError(error_msg))
        span.end()
