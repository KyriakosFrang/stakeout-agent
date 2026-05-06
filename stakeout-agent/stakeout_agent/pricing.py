from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPricing:
    input_cost_per_1k: float
    output_cost_per_1k: float


class PricingMap:
    """Opt-in cost estimator. Pass an instance to LangGraphMonitorCallback to enable cost tracking.

    Example::

        pricing = PricingMap({
            "gpt-4o": ModelPricing(input_cost_per_1k=0.005, output_cost_per_1k=0.015),
        })
    """

    def __init__(self, prices: dict[str, ModelPricing]) -> None:
        self._prices = prices

    def estimate_cost(self, model: str | None, input_tokens: int, output_tokens: int) -> float | None:
        if model is None:
            return None
        pricing = self._prices.get(model)
        if pricing is None:
            return None
        return (input_tokens / 1000 * pricing.input_cost_per_1k) + (output_tokens / 1000 * pricing.output_cost_per_1k)
