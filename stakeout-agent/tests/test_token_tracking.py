from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from langchain_core.outputs import LLMResult

from stakeout_agent.callback_handler import AsyncLangGraphMonitorCallback, LangGraphMonitorCallback
from stakeout_agent.pricing import ModelPricing, PricingMap

GRAPH_ID = "test_graph"
THREAD_ID = "thread_1"


def make_uuid():
    return uuid4()


def mock_db() -> MagicMock:
    return MagicMock()


def _make_llm_result(token_usage: dict, model_name: str = "gpt-4o") -> LLMResult:
    return LLMResult(generations=[], llm_output={"token_usage": token_usage, "model_name": model_name})


def _make_anthropic_result(usage: dict, model: str = "claude-3-5-sonnet") -> LLMResult:
    return LLMResult(generations=[], llm_output={"usage": usage, "model": model})


# ---------------------------------------------------------------------------
# PricingMap unit tests
# ---------------------------------------------------------------------------


class TestPricingMap:
    def _map(self):
        return PricingMap(
            {
                "gpt-4o": ModelPricing(input_cost_per_1k=0.005, output_cost_per_1k=0.015),
                "gpt-4o-mini": ModelPricing(input_cost_per_1k=0.00015, output_cost_per_1k=0.0006),
            }
        )

    def test_known_model_returns_cost(self):
        pricing = self._map()
        cost = pricing.estimate_cost("gpt-4o", 1000, 500)
        assert cost == pytest.approx(0.005 * 1 + 0.015 * 0.5)

    def test_unknown_model_returns_none(self):
        pricing = self._map()
        assert pricing.estimate_cost("unknown-model", 1000, 500) is None

    def test_none_model_returns_none(self):
        pricing = self._map()
        assert pricing.estimate_cost(None, 1000, 500) is None

    def test_zero_tokens_returns_zero_cost(self):
        pricing = self._map()
        assert pricing.estimate_cost("gpt-4o", 0, 0) == 0.0


# ---------------------------------------------------------------------------
# Default token extractor
# ---------------------------------------------------------------------------


class TestDefaultTokenExtractor:
    def _cb(self):
        return LangGraphMonitorCallback(graph_id=GRAPH_ID, thread_id=THREAD_ID, db=mock_db())

    def test_openai_format(self):
        cb = self._cb()
        metadata = {"token_usage": {"prompt_tokens": 100, "completion_tokens": 50}, "model_name": "gpt-4o"}
        in_tok, out_tok, model = cb._token_extractor(metadata)
        assert in_tok == 100
        assert out_tok == 50
        assert model == "gpt-4o"

    def test_anthropic_format(self):
        cb = self._cb()
        metadata = {"usage": {"input_tokens": 200, "output_tokens": 80}, "model": "claude-3-5-sonnet"}
        in_tok, out_tok, model = cb._token_extractor(metadata)
        assert in_tok == 200
        assert out_tok == 80
        assert model == "claude-3-5-sonnet"

    def test_unknown_format_returns_nones(self):
        cb = self._cb()
        in_tok, out_tok, model = cb._token_extractor({})
        assert in_tok is None
        assert out_tok is None
        assert model is None

    def test_custom_extractor_is_used(self):
        def my_extractor(meta):
            return meta.get("in"), meta.get("out"), meta.get("m")

        cb = LangGraphMonitorCallback(
            graph_id=GRAPH_ID, thread_id=THREAD_ID, db=mock_db(), token_extractor=my_extractor
        )
        in_tok, out_tok, model = cb._token_extractor({"in": 10, "out": 5, "m": "my-model"})
        assert in_tok == 10
        assert out_tok == 5
        assert model == "my-model"


# ---------------------------------------------------------------------------
# Token accumulation on node_end events
# ---------------------------------------------------------------------------


class TestTokenAccumulation:
    def _make(self, pricing=None):
        db = mock_db()
        cb = LangGraphMonitorCallback(graph_id=GRAPH_ID, thread_id=THREAD_ID, db=db, pricing=pricing)
        return cb, db

    def test_on_llm_end_accumulates_tokens_on_node_end(self):
        cb, db = self._make()
        root_id = make_uuid()
        node_id = make_uuid()
        llm_id = make_uuid()

        cb.on_chain_start({}, {}, run_id=root_id, parent_run_id=None)
        cb.on_chain_start({"name": "llm_node"}, {}, run_id=node_id, parent_run_id=root_id)
        response = _make_llm_result({"prompt_tokens": 100, "completion_tokens": 50})
        cb.on_llm_end(response, run_id=llm_id, parent_run_id=node_id)
        cb.on_chain_end({}, run_id=node_id, parent_run_id=root_id)

        node_end_call = db.insert_event.call_args_list[-1]
        assert node_end_call.kwargs["input_tokens"] == 100
        assert node_end_call.kwargs["output_tokens"] == 50
        assert node_end_call.kwargs["model"] == "gpt-4o"

    def test_multiple_llm_calls_summed_per_node(self):
        cb, db = self._make()
        root_id = make_uuid()
        node_id = make_uuid()

        cb.on_chain_start({}, {}, run_id=root_id, parent_run_id=None)
        cb.on_chain_start({"name": "n"}, {}, run_id=node_id, parent_run_id=root_id)
        cb.on_llm_end(
            _make_llm_result({"prompt_tokens": 100, "completion_tokens": 40}), run_id=make_uuid(), parent_run_id=node_id
        )
        cb.on_llm_end(
            _make_llm_result({"prompt_tokens": 200, "completion_tokens": 60}), run_id=make_uuid(), parent_run_id=node_id
        )
        cb.on_chain_end({}, run_id=node_id, parent_run_id=root_id)

        node_end_call = db.insert_event.call_args_list[-1]
        assert node_end_call.kwargs["input_tokens"] == 300
        assert node_end_call.kwargs["output_tokens"] == 100

    def test_no_llm_calls_leaves_tokens_none_on_node_end(self):
        cb, db = self._make()
        root_id = make_uuid()
        node_id = make_uuid()

        cb.on_chain_start({}, {}, run_id=root_id, parent_run_id=None)
        cb.on_chain_start({"name": "n"}, {}, run_id=node_id, parent_run_id=root_id)
        cb.on_chain_end({}, run_id=node_id, parent_run_id=root_id)

        node_end_call = db.insert_event.call_args_list[-1]
        assert node_end_call.kwargs["input_tokens"] is None
        assert node_end_call.kwargs["output_tokens"] is None
        assert node_end_call.kwargs["model"] is None

    def test_run_totals_passed_to_complete_run(self):
        cb, db = self._make()
        root_id = make_uuid()
        node_id = make_uuid()

        cb.on_chain_start({}, {}, run_id=root_id, parent_run_id=None)
        cb.on_chain_start({"name": "n"}, {}, run_id=node_id, parent_run_id=root_id)
        cb.on_llm_end(
            _make_llm_result({"prompt_tokens": 300, "completion_tokens": 120}),
            run_id=make_uuid(),
            parent_run_id=node_id,
        )
        cb.on_chain_end({}, run_id=node_id, parent_run_id=root_id)
        cb.on_chain_end({}, run_id=root_id, parent_run_id=None)

        kw = db.complete_run.call_args.kwargs
        assert kw["total_input_tokens"] == 300
        assert kw["total_output_tokens"] == 120
        assert kw["estimated_cost_usd"] is None

    def test_no_tokens_passes_none_totals_to_complete_run(self):
        cb, db = self._make()
        root_id = make_uuid()

        cb.on_chain_start({}, {}, run_id=root_id, parent_run_id=None)
        cb.on_chain_end({}, run_id=root_id, parent_run_id=None)

        kw = db.complete_run.call_args.kwargs
        assert kw["total_input_tokens"] is None
        assert kw["total_output_tokens"] is None
        assert kw["estimated_cost_usd"] is None

    def test_state_cleared_after_run(self):
        cb, db = self._make()
        root_id = make_uuid()
        node_id = make_uuid()

        cb.on_chain_start({}, {}, run_id=root_id, parent_run_id=None)
        cb.on_chain_start({"name": "n"}, {}, run_id=node_id, parent_run_id=root_id)
        cb.on_llm_end(
            _make_llm_result({"prompt_tokens": 100, "completion_tokens": 50}), run_id=make_uuid(), parent_run_id=node_id
        )
        cb.on_chain_end({}, run_id=node_id, parent_run_id=root_id)
        cb.on_chain_end({}, run_id=root_id, parent_run_id=None)

        assert cb._total_input_tokens == 0
        assert cb._total_output_tokens == 0
        assert cb._total_cost is None
        assert cb._node_tokens == {}


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------


class TestCostEstimation:
    def _pricing(self):
        return PricingMap(
            {
                "gpt-4o": ModelPricing(input_cost_per_1k=0.005, output_cost_per_1k=0.015),
                "gpt-4o-mini": ModelPricing(input_cost_per_1k=0.00015, output_cost_per_1k=0.0006),
            }
        )

    def _make(self, pricing=None):
        db = mock_db()
        cb = LangGraphMonitorCallback(graph_id=GRAPH_ID, thread_id=THREAD_ID, db=db, pricing=pricing)
        return cb, db

    def test_cost_estimated_when_pricing_configured(self):
        cb, db = self._make(pricing=self._pricing())
        root_id = make_uuid()
        node_id = make_uuid()

        cb.on_chain_start({}, {}, run_id=root_id, parent_run_id=None)
        cb.on_chain_start({"name": "n"}, {}, run_id=node_id, parent_run_id=root_id)
        cb.on_llm_end(
            _make_llm_result({"prompt_tokens": 1000, "completion_tokens": 500}),
            run_id=make_uuid(),
            parent_run_id=node_id,
        )
        cb.on_chain_end({}, run_id=node_id, parent_run_id=root_id)
        cb.on_chain_end({}, run_id=root_id, parent_run_id=None)

        kw = db.complete_run.call_args.kwargs
        expected = (1000 / 1000 * 0.005) + (500 / 1000 * 0.015)
        assert kw["estimated_cost_usd"] == pytest.approx(expected)

    def test_multi_model_cost_summed(self):
        cb, db = self._make(pricing=self._pricing())
        root_id = make_uuid()
        node1_id = make_uuid()
        node2_id = make_uuid()

        cb.on_chain_start({}, {}, run_id=root_id, parent_run_id=None)

        cb.on_chain_start({"name": "n1"}, {}, run_id=node1_id, parent_run_id=root_id)
        cb.on_llm_end(
            _make_llm_result({"prompt_tokens": 1000, "completion_tokens": 500}, "gpt-4o"),
            run_id=make_uuid(),
            parent_run_id=node1_id,
        )
        cb.on_chain_end({}, run_id=node1_id, parent_run_id=root_id)

        cb.on_chain_start({"name": "n2"}, {}, run_id=node2_id, parent_run_id=root_id)
        cb.on_llm_end(
            _make_llm_result({"prompt_tokens": 2000, "completion_tokens": 1000}, "gpt-4o-mini"),
            run_id=make_uuid(),
            parent_run_id=node2_id,
        )
        cb.on_chain_end({}, run_id=node2_id, parent_run_id=root_id)

        cb.on_chain_end({}, run_id=root_id, parent_run_id=None)

        cost_4o = (1000 / 1000 * 0.005) + (500 / 1000 * 0.015)
        cost_mini = (2000 / 1000 * 0.00015) + (1000 / 1000 * 0.0006)
        kw = db.complete_run.call_args.kwargs
        assert kw["estimated_cost_usd"] == pytest.approx(cost_4o + cost_mini)
        assert kw["total_input_tokens"] == 3000
        assert kw["total_output_tokens"] == 1500

    def test_unknown_model_excluded_from_cost(self):
        cb, db = self._make(pricing=self._pricing())
        root_id = make_uuid()
        node_id = make_uuid()

        cb.on_chain_start({}, {}, run_id=root_id, parent_run_id=None)
        cb.on_chain_start({"name": "n"}, {}, run_id=node_id, parent_run_id=root_id)
        cb.on_llm_end(
            _make_llm_result({"prompt_tokens": 500, "completion_tokens": 200}, "unknown-model"),
            run_id=make_uuid(),
            parent_run_id=node_id,
        )
        cb.on_chain_end({}, run_id=node_id, parent_run_id=root_id)
        cb.on_chain_end({}, run_id=root_id, parent_run_id=None)

        kw = db.complete_run.call_args.kwargs
        assert kw["estimated_cost_usd"] is None
        assert kw["total_input_tokens"] == 500

    def test_no_pricing_means_no_cost(self):
        cb, db = self._make(pricing=None)
        root_id = make_uuid()
        node_id = make_uuid()

        cb.on_chain_start({}, {}, run_id=root_id, parent_run_id=None)
        cb.on_chain_start({"name": "n"}, {}, run_id=node_id, parent_run_id=root_id)
        cb.on_llm_end(
            _make_llm_result({"prompt_tokens": 100, "completion_tokens": 50}), run_id=make_uuid(), parent_run_id=node_id
        )
        cb.on_chain_end({}, run_id=node_id, parent_run_id=root_id)
        cb.on_chain_end({}, run_id=root_id, parent_run_id=None)

        kw = db.complete_run.call_args.kwargs
        assert kw["estimated_cost_usd"] is None
        assert kw["total_input_tokens"] == 100


# ---------------------------------------------------------------------------
# Async callback token tracking
# ---------------------------------------------------------------------------


class TestAsyncTokenTracking:
    def _make(self, pricing=None):
        db = mock_db()
        cb = AsyncLangGraphMonitorCallback(graph_id=GRAPH_ID, thread_id=THREAD_ID, db=db, pricing=pricing)
        return cb, db

    async def test_on_llm_end_accumulates_tokens(self):
        cb, db = self._make()
        root_id = make_uuid()
        node_id = make_uuid()
        llm_id = make_uuid()

        await cb.on_chain_start({}, {}, run_id=root_id, parent_run_id=None)
        await cb.on_chain_start({"name": "n"}, {}, run_id=node_id, parent_run_id=root_id)
        response = _make_llm_result({"prompt_tokens": 150, "completion_tokens": 75})
        await cb.on_llm_end(response, run_id=llm_id, parent_run_id=node_id)
        await cb.on_chain_end({}, run_id=node_id, parent_run_id=root_id)
        await cb.on_chain_end({}, run_id=root_id, parent_run_id=None)

        kw = db.complete_run.call_args.kwargs
        assert kw["total_input_tokens"] == 150
        assert kw["total_output_tokens"] == 75

    async def test_cost_estimated_async(self):
        pricing = PricingMap({"gpt-4o": ModelPricing(input_cost_per_1k=0.005, output_cost_per_1k=0.015)})
        cb, db = self._make(pricing=pricing)
        root_id = make_uuid()
        node_id = make_uuid()

        await cb.on_chain_start({}, {}, run_id=root_id, parent_run_id=None)
        await cb.on_chain_start({"name": "n"}, {}, run_id=node_id, parent_run_id=root_id)
        await cb.on_llm_end(
            _make_llm_result({"prompt_tokens": 1000, "completion_tokens": 500}),
            run_id=make_uuid(),
            parent_run_id=node_id,
        )
        await cb.on_chain_end({}, run_id=node_id, parent_run_id=root_id)
        await cb.on_chain_end({}, run_id=root_id, parent_run_id=None)

        kw = db.complete_run.call_args.kwargs
        expected = (1000 / 1000 * 0.005) + (500 / 1000 * 0.015)
        assert kw["estimated_cost_usd"] == pytest.approx(expected)
