from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from typing import Any
from uuid import UUID

from stakeout_agent.backends.base import AbstractMonitorDB

_logger = logging.getLogger(__name__)


def _extract_cache_tokens(metadata: dict) -> tuple[int | None, int | None]:
    """Extract (cache_read_tokens, cache_creation_tokens) from LLM response metadata.

    Covers OpenAI (prompt_tokens_details.cached_tokens) and Anthropic
    (usage.cache_read_input_tokens / cache_creation_input_tokens) conventions.
    """
    # OpenAI: token_usage.prompt_tokens_details.cached_tokens
    token_usage = metadata.get("token_usage") or {}
    if token_usage:
        details = token_usage.get("prompt_tokens_details") or {}
        cached = details.get("cached_tokens")
        if cached is not None:
            return cached, None
    # Anthropic: usage.cache_read_input_tokens + cache_creation_input_tokens
    usage = metadata.get("usage") or {}
    if usage:
        cache_read = usage.get("cache_read_input_tokens")
        cache_creation = usage.get("cache_creation_input_tokens")
        if cache_read is not None or cache_creation is not None:
            return cache_read, cache_creation
    return None, None


def _default_token_extractor(metadata: dict) -> tuple[int | None, int | None, str | None]:
    """Extract (input_tokens, output_tokens, model) from LLM response metadata.

    Covers OpenAI (token_usage / model_name) and Anthropic (usage / model) conventions.
    """
    # OpenAI: llm_output["token_usage"] + model_name
    usage = metadata.get("token_usage") or {}
    if usage:
        return usage.get("prompt_tokens"), usage.get("completion_tokens"), metadata.get("model_name")
    # Anthropic: llm_output["usage"] + model
    usage = metadata.get("usage") or {}
    if usage:
        return usage.get("input_tokens"), usage.get("output_tokens"), metadata.get("model")
    return None, None, None


class _MonitorBase:
    """Shared state and logic reused by all framework-specific callback handlers."""

    def __init__(
        self,
        graph_id: str,
        thread_id: str,
        db: AbstractMonitorDB | None = None,
        pricing=None,
        token_extractor: Callable[[dict], tuple[int | None, int | None, str | None]] | None = None,
        capture_payloads: bool = True,
        max_payload_chars: int | None = None,
    ):
        self.graph_id = graph_id
        self.thread_id = thread_id
        if db is None:
            from stakeout_agent.backends import get_backend

            db = get_backend()
        self.db = db
        self._pricing = pricing
        self._token_extractor = token_extractor or _default_token_extractor
        self.capture_payloads = capture_payloads
        self._max_payload_chars = max_payload_chars
        self._log = logging.LoggerAdapter(_logger, {"graph_id": graph_id, "thread_id": thread_id})

        self._state_lock = threading.Lock()
        self._run_id: str | None = None
        self._node_start_times: dict[str, float] = {}
        self._node_names: dict[str, str] = {}
        self._tool_start_times: dict[str, float] = {}
        self._retriever_names: dict[str, str] = {}
        # Per-node token accumulation keyed by node run_id
        self._node_tokens: dict[str, dict] = {}
        # Per-node LLM prompt/response capture keyed by node run_id
        self._llm_inputs: dict[str, list[dict]] = {}
        self._llm_outputs: dict[str, str] = {}
        # Run-level token totals
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._total_cache_read_tokens: int = 0
        self._total_cache_creation_tokens: int = 0
        # None until first successful cost estimate; avoids confusing 0.0 with "not configured"
        self._total_cost: float | None = None
        # Cumulative count of DB writes that raised an exception
        self._dropped_events: int = 0

    @property
    def dropped_events(self) -> int:
        """Number of DB writes that failed since this monitor was created."""
        with self._state_lock:
            return self._dropped_events

    def _safe_db_write(self, fn: Callable[[], Any]) -> None:
        try:
            fn()
        except Exception as exc:
            with self._state_lock:
                self._dropped_events += 1
                count = self._dropped_events
            self._log.warning("DB write dropped (total dropped: %d): %s", count, exc)

    def _handle_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: dict[str, Any],
        run_id: UUID,
        parent_run_id: UUID | None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        run_id_str = str(run_id)
        if parent_run_id is None:
            with self._state_lock:
                self._run_id = run_id_str
            self._log.debug("run started run_id=%s", run_id_str)
            self._safe_db_write(lambda: self.db.create_run(run_id_str, self.graph_id, self.thread_id))
        else:
            node_name = self._extract_name(serialized, kwargs)
            with self._state_lock:
                self._node_start_times[run_id_str] = time.monotonic()
                self._node_names[run_id_str] = node_name
                current_run_id = self._run_id
            self._log.debug("node_start node=%s run_id=%s", node_name, current_run_id)
            payload: dict[str, Any] = {"inputs": self._safe_truncate(inputs)}
            metadata = kwargs.get("metadata")
            if metadata:
                payload["metadata"] = metadata
            if tags:
                payload["tags"] = tags
            messages = self._extract_messages(inputs)
            self._safe_db_write(
                lambda: self.db.insert_event(
                    run_id=current_run_id,
                    graph_id=self.graph_id,
                    event_type="node_start",
                    node_name=node_name,
                    payload=payload,
                    messages=messages,
                )
            )

    def _handle_chain_end(
        self,
        outputs: dict[str, Any],
        run_id: UUID,
        parent_run_id: UUID | None,
        **kwargs: Any,
    ) -> None:
        run_id_str = str(run_id)
        if parent_run_id is None:
            with self._state_lock:
                current_run_id = self._run_id
                total_in = self._total_input_tokens or None
                total_out = self._total_output_tokens or None
                total_cr = self._total_cache_read_tokens or None
                total_cc = self._total_cache_creation_tokens or None
                cost = self._total_cost
                self._clear_timing_state()
            self._log.debug("run completed run_id=%s", current_run_id)
            self._safe_db_write(
                lambda: self.db.complete_run(
                    current_run_id,
                    total_input_tokens=total_in,
                    total_output_tokens=total_out,
                    estimated_cost_usd=cost,
                    total_cache_read_tokens=total_cr,
                    total_cache_creation_tokens=total_cc,
                )
            )
        else:
            with self._state_lock:
                latency = self._pop_latency(self._node_start_times, run_id_str)
                node_name = self._node_names.pop(run_id_str, "unknown")
                current_run_id = self._run_id
                tok = self._node_tokens.pop(run_id_str, {})
                llm_input = self._llm_inputs.pop(run_id_str, None) if self.capture_payloads else None
                llm_output = self._llm_outputs.pop(run_id_str, None) if self.capture_payloads else None
            input_tokens = tok.get("input") or None
            output_tokens = tok.get("output") or None
            cache_read_tokens = tok.get("cache_read") or None
            cache_creation_tokens = tok.get("cache_creation") or None
            model = tok.get("model")
            self._log.debug("node_end node=%s latency_ms=%s run_id=%s", node_name, latency, current_run_id)
            payload = {"outputs": self._safe_truncate(outputs)}
            messages = self._extract_messages(outputs)
            self._safe_db_write(
                lambda: self.db.insert_event(
                    run_id=current_run_id,
                    graph_id=self.graph_id,
                    event_type="node_end",
                    node_name=node_name,
                    latency_ms=latency,
                    payload=payload,
                    messages=messages,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model=model,
                    llm_input=llm_input,
                    llm_output=llm_output,
                    cache_read_tokens=cache_read_tokens,
                    cache_creation_tokens=cache_creation_tokens,
                )
            )

    def _handle_chain_error(
        self,
        error: BaseException,
        run_id: UUID,
        parent_run_id: UUID | None,
        **kwargs: Any,
    ) -> None:
        run_id_str = str(run_id)
        error_str = f"{type(error).__name__}: {str(error)}"
        if parent_run_id is None:
            with self._state_lock:
                current_run_id = self._run_id
                self._clear_timing_state()
            self._log.warning("run failed run_id=%s error=%s", current_run_id, error_str)
            self._safe_db_write(lambda: self.db.fail_run(current_run_id, error_str))
        else:
            with self._state_lock:
                latency = self._pop_latency(self._node_start_times, run_id_str)
                node_name = self._node_names.pop(run_id_str, "unknown")
                current_run_id = self._run_id
            self._log.warning("node error node=%s run_id=%s error=%s", node_name, current_run_id, error_str)
            self._safe_db_write(
                lambda: self.db.insert_event(
                    run_id=current_run_id,
                    graph_id=self.graph_id,
                    event_type="error",
                    node_name=node_name,
                    latency_ms=latency,
                    error=error_str,
                )
            )

    def _handle_tool_start(
        self,
        serialized: dict[str, Any] | None,
        input_str: str,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        run_id_str = str(run_id)
        tool_name = serialized.get("name", "unknown_tool") if serialized else kwargs.get("name", "unknown_tool")
        with self._state_lock:
            self._tool_start_times[run_id_str] = time.monotonic()
            current_run_id = self._run_id
        self._log.debug("tool_call tool=%s run_id=%s", tool_name, current_run_id)
        structured_inputs = kwargs.get("inputs")
        raw_input = self._safe_truncate(structured_inputs) if structured_inputs is not None else input_str[:500]
        self._safe_db_write(
            lambda: self.db.insert_event(
                run_id=current_run_id,
                graph_id=self.graph_id,
                event_type="tool_call",
                node_name=tool_name,
                payload={"input": raw_input},
            )
        )

    def _handle_tool_end(self, output: Any, run_id: UUID, **kwargs: Any) -> None:
        run_id_str = str(run_id)
        tool_name = kwargs.get("name", "unknown_tool")
        with self._state_lock:
            latency = self._pop_latency(self._tool_start_times, run_id_str)
            current_run_id = self._run_id
        self._log.debug("tool_result tool=%s latency_ms=%s run_id=%s", tool_name, latency, current_run_id)
        truncated_output = str(output)[:500]
        self._safe_db_write(
            lambda: self.db.insert_event(
                run_id=current_run_id,
                graph_id=self.graph_id,
                event_type="tool_result",
                node_name=tool_name,
                latency_ms=latency,
                payload={"output": truncated_output},
            )
        )

    def _handle_tool_error(self, error: BaseException, run_id: UUID, **kwargs: Any) -> None:
        run_id_str = str(run_id)
        tool_name = kwargs.get("name", "unknown_tool")
        with self._state_lock:
            latency = self._pop_latency(self._tool_start_times, run_id_str)
            current_run_id = self._run_id
        error_str = f"{type(error).__name__}: {str(error)}"
        self._log.warning("tool error tool=%s run_id=%s error=%s", tool_name, current_run_id, error_str)
        self._safe_db_write(
            lambda: self.db.insert_event(
                run_id=current_run_id,
                graph_id=self.graph_id,
                event_type="error",
                node_name=tool_name,
                latency_ms=latency,
                error=error_str,
            )
        )

    def _handle_retriever_start(
        self,
        serialized: dict[str, Any] | None,
        query: str,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        run_id_str = str(run_id)
        retriever_name = serialized.get("id", ["unknown_retriever"])[-1] if serialized else "unknown_retriever"
        with self._state_lock:
            self._tool_start_times[run_id_str] = time.monotonic()
            self._retriever_names[run_id_str] = retriever_name
            current_run_id = self._run_id
        self._log.debug("retriever_start name=%s run_id=%s", retriever_name, current_run_id)
        self._safe_db_write(
            lambda: self.db.insert_event(
                run_id=current_run_id,
                graph_id=self.graph_id,
                event_type="retriever_start",
                node_name=retriever_name,
                payload={"query": query[:500]},
            )
        )

    def _handle_retriever_end(self, documents: Any, run_id: UUID, **kwargs: Any) -> None:
        run_id_str = str(run_id)
        with self._state_lock:
            latency = self._pop_latency(self._tool_start_times, run_id_str)
            retriever_name = self._retriever_names.pop(run_id_str, "unknown_retriever")
            current_run_id = self._run_id
        doc_count = len(documents) if documents is not None else 0
        self._log.debug(
            "retriever_end name=%s docs=%d latency_ms=%s run_id=%s", retriever_name, doc_count, latency, current_run_id
        )
        self._safe_db_write(
            lambda: self.db.insert_event(
                run_id=current_run_id,
                graph_id=self.graph_id,
                event_type="retriever_end",
                node_name=retriever_name,
                latency_ms=latency,
                payload={"document_count": doc_count},
            )
        )

    def _handle_retriever_error(self, error: BaseException, run_id: UUID, **kwargs: Any) -> None:
        run_id_str = str(run_id)
        with self._state_lock:
            latency = self._pop_latency(self._tool_start_times, run_id_str)
            retriever_name = self._retriever_names.pop(run_id_str, "unknown_retriever")
            current_run_id = self._run_id
        error_str = f"{type(error).__name__}: {str(error)}"
        self._log.warning("retriever error name=%s run_id=%s error=%s", retriever_name, current_run_id, error_str)
        self._safe_db_write(
            lambda: self.db.insert_event(
                run_id=current_run_id,
                graph_id=self.graph_id,
                event_type="error",
                node_name=retriever_name,
                latency_ms=latency,
                error=error_str,
            )
        )

    def _handle_llm_start(self, formatted_messages: list[dict], parent_run_id: UUID | None) -> None:
        """Store LLM prompt messages for the enclosing node, keyed by the node's run_id."""
        if not self.capture_payloads or parent_run_id is None:
            return
        if self._max_payload_chars is not None:
            formatted_messages = [
                {"role": m["role"], "content": m["content"][: self._max_payload_chars]} for m in formatted_messages
            ]
        key = str(parent_run_id)
        with self._state_lock:
            self._llm_inputs.setdefault(key, []).extend(formatted_messages)

    def _handle_llm_end(self, metadata: dict, parent_run_id: UUID | None, output_text: str | None = None) -> None:
        """Accumulate token counts and capture LLM output text from a completed LLM call."""
        input_tok, output_tok, model = self._token_extractor(metadata)
        cache_read_tok, cache_creation_tok = _extract_cache_tokens(metadata)

        in_tok = input_tok or 0
        out_tok = output_tok or 0
        cr_tok = cache_read_tok or 0
        cc_tok = cache_creation_tok or 0

        with self._state_lock:
            if input_tok is not None or output_tok is not None:
                self._total_input_tokens += in_tok
                self._total_output_tokens += out_tok

                if parent_run_id is not None:
                    key = str(parent_run_id)
                    entry = self._node_tokens.setdefault(
                        key, {"input": 0, "output": 0, "cache_read": 0, "cache_creation": 0, "model": None}
                    )
                    entry["input"] += in_tok
                    entry["output"] += out_tok
                    if model:
                        entry["model"] = model

            if cache_read_tok is not None or cache_creation_tok is not None:
                self._total_cache_read_tokens += cr_tok
                self._total_cache_creation_tokens += cc_tok

                if parent_run_id is not None:
                    key = str(parent_run_id)
                    entry = self._node_tokens.setdefault(
                        key, {"input": 0, "output": 0, "cache_read": 0, "cache_creation": 0, "model": None}
                    )
                    entry["cache_read"] += cr_tok
                    entry["cache_creation"] += cc_tok

            if self.capture_payloads and parent_run_id is not None and output_text is not None:
                if self._max_payload_chars is not None:
                    output_text = output_text[: self._max_payload_chars]
                self._llm_outputs[str(parent_run_id)] = output_text

        if self._pricing is not None and (input_tok is not None or output_tok is not None):
            cost = self._pricing.estimate_cost(model, in_tok, out_tok)
            if cost is not None:
                with self._state_lock:
                    self._total_cost = (self._total_cost or 0.0) + cost

    def _clear_timing_state(self) -> None:
        """Clear all per-run state. Must be called under _state_lock."""
        self._node_start_times.clear()
        self._node_names.clear()
        self._tool_start_times.clear()
        self._retriever_names.clear()
        self._node_tokens.clear()
        self._llm_inputs.clear()
        self._llm_outputs.clear()
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_cache_read_tokens = 0
        self._total_cache_creation_tokens = 0
        self._total_cost = None
        self._run_id = None

    @staticmethod
    def _extract_messages(data: Any) -> list[dict] | None:
        """Extract a messages list from a LangGraph state dict into plain {role, content} dicts.

        Returns None when the state has no messages field, so callers can omit the field entirely.
        Handles both LangChain BaseMessage objects and already-serialised dicts.
        """
        if not isinstance(data, dict):
            return None
        msgs = data.get("messages")
        if not isinstance(msgs, list) or not msgs:
            return None
        _ROLE_MAP = {"human": "human", "ai": "assistant", "system": "system", "tool": "tool"}
        result = []
        for m in msgs:
            if hasattr(m, "type") and hasattr(m, "content"):
                # LangChain BaseMessage subclass
                role = _ROLE_MAP.get(m.type, m.type)
                content = m.content if isinstance(m.content, str) else str(m.content)
                result.append({"role": role, "content": content[:500]})
            elif isinstance(m, dict) and "role" in m:
                result.append({"role": m["role"], "content": str(m.get("content", ""))[:500]})
        return result or None

    @staticmethod
    def _extract_name(serialized: dict | None, kwargs: dict) -> str:
        if serialized:
            ids = serialized.get("id") or ["unknown"]
            return serialized.get("name") or ids[-1] or kwargs.get("name", "unknown")
        return kwargs.get("name", "unknown")

    @staticmethod
    def _pop_latency(store: dict[str, float], key: str) -> float | None:
        start = store.pop(key, None)
        if start is None:
            return None
        return round((time.monotonic() - start) * 1000, 2)

    @staticmethod
    def _safe_truncate(data: Any, max_len: int = 500) -> str:
        if isinstance(data, str):
            return data[:max_len]
        try:
            text = json.dumps(data, default=str)
        except Exception:
            try:
                text = str(data)
            except Exception:
                return ""
        return text[:max_len]
