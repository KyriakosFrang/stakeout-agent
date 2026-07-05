from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from stakeout_agent.alerts import AlertManager, Rule, _RunSample

# ---------------------------------------------------------------------------
# _compute unit tests
# ---------------------------------------------------------------------------


class TestCompute:
    def test_error_rate_all_failed(self):
        samples = [_RunSample(0, "failed", None, None)] * 3
        assert AlertManager._compute("error_rate", samples) == pytest.approx(1.0)

    def test_error_rate_mixed(self):
        samples = [
            _RunSample(0, "failed", None, None),
            _RunSample(0, "completed", None, None),
            _RunSample(0, "completed", None, None),
            _RunSample(0, "completed", None, None),
        ]
        assert AlertManager._compute("error_rate", samples) == pytest.approx(0.25)

    def test_error_rate_empty_window(self):
        assert AlertManager._compute("error_rate", []) is None

    def test_p95_latency_single_sample(self):
        samples = [_RunSample(0, "completed", 500.0, None)]
        assert AlertManager._compute("p95_latency_ms", samples) == pytest.approx(500.0)

    def test_p95_latency_multiple(self):
        import statistics

        latencies = list(range(1, 101))  # 1..100
        samples = [_RunSample(0, "completed", float(v), None) for v in latencies]
        result = AlertManager._compute("p95_latency_ms", samples)
        expected = statistics.quantiles([float(v) for v in latencies], n=100)[94]
        assert result == pytest.approx(expected)

    def test_p99_latency_multiple(self):
        import statistics

        latencies = list(range(1, 101))
        samples = [_RunSample(0, "completed", float(v), None) for v in latencies]
        result = AlertManager._compute("p99_latency_ms", samples)
        expected = statistics.quantiles([float(v) for v in latencies], n=100)[98]
        assert result == pytest.approx(expected)

    def test_p95_no_latency_data(self):
        samples = [_RunSample(0, "completed", None, None)] * 5
        assert AlertManager._compute("p95_latency_ms", samples) is None

    def test_estimated_cost_usd(self):
        samples = [
            _RunSample(0, "completed", None, 1.0),
            _RunSample(0, "completed", None, 2.5),
            _RunSample(0, "completed", None, None),
        ]
        assert AlertManager._compute("estimated_cost_usd", samples) == pytest.approx(3.5)

    def test_estimated_cost_usd_no_cost_data(self):
        samples = [_RunSample(0, "completed", None, None)] * 3
        assert AlertManager._compute("estimated_cost_usd", samples) is None

    def test_unknown_metric(self):
        samples = [_RunSample(0, "completed", 100.0, 1.0)]
        assert AlertManager._compute("nonexistent_metric", samples) is None


# ---------------------------------------------------------------------------
# AlertManager integration tests (mocked HTTP)
# ---------------------------------------------------------------------------


def _make_manager(**kwargs) -> AlertManager:
    defaults = {
        "rules": [Rule(metric="error_rate", window_seconds=300, threshold=0.05)],
        "webhook_url": "https://hooks.example.com/test",
        "cooldown_seconds": 300,
    }
    defaults.update(kwargs)
    return AlertManager(**defaults)


class TestThresholdBreach:
    def test_fires_when_threshold_exceeded(self):
        manager = _make_manager()
        with patch("urllib.request.urlopen") as mock_open:
            mock_open.return_value.__enter__ = lambda s: s
            mock_open.return_value.__exit__ = MagicMock(return_value=False)
            # 100% error rate → exceeds 5%
            manager.record_and_evaluate(status="failed", latency_ms=100.0, cost=None, graph_id="g1")
            manager.close(wait=True)
        mock_open.assert_called_once()
        req = mock_open.call_args[0][0]
        payload = json.loads(req.data)
        assert payload["alert"] == "error_rate"
        assert payload["value"] == pytest.approx(1.0)
        assert payload["threshold"] == pytest.approx(0.05)
        assert payload["graph_id"] == "g1"
        assert "fired_at" in payload

    def test_no_fire_below_threshold(self):
        manager = _make_manager()
        with patch("urllib.request.urlopen") as mock_open:
            mock_open.return_value.__enter__ = lambda s: s
            mock_open.return_value.__exit__ = MagicMock(return_value=False)
            # 0% error rate → below 5%
            manager.record_and_evaluate(status="completed", latency_ms=100.0, cost=None, graph_id="g1")
        mock_open.assert_not_called()

    def test_no_fire_exactly_at_threshold(self):
        # 1 completed then 1 failed → error_rate = 0.5 = threshold → not strictly greater, no fire
        manager = _make_manager(rules=[Rule(metric="error_rate", window_seconds=300, threshold=0.5)])
        with patch("urllib.request.urlopen") as mock_open:
            mock_open.return_value.__enter__ = lambda s: s
            mock_open.return_value.__exit__ = MagicMock(return_value=False)
            manager.record_and_evaluate(status="completed", latency_ms=None, cost=None, graph_id="g1")
            manager.record_and_evaluate(status="failed", latency_ms=None, cost=None, graph_id="g1")
        mock_open.assert_not_called()


class TestCooldown:
    def test_cooldown_suppresses_second_fire(self):
        manager = _make_manager(cooldown_seconds=60)
        with patch("urllib.request.urlopen") as mock_open:
            mock_open.return_value.__enter__ = lambda s: s
            mock_open.return_value.__exit__ = MagicMock(return_value=False)
            manager.record_and_evaluate(status="failed", latency_ms=None, cost=None, graph_id="g1")
            manager.record_and_evaluate(status="failed", latency_ms=None, cost=None, graph_id="g1")
            manager.close(wait=True)
        # Only one webhook call despite two breaches
        assert mock_open.call_count == 1

    def test_fires_again_after_cooldown_expires(self):
        manager = _make_manager(cooldown_seconds=1)
        with patch("urllib.request.urlopen") as mock_open:
            mock_open.return_value.__enter__ = lambda s: s
            mock_open.return_value.__exit__ = MagicMock(return_value=False)
            manager.record_and_evaluate(status="failed", latency_ms=None, cost=None, graph_id="g1")
            # Backdate the last_fired time to simulate cooldown expiry
            manager._last_fired["error_rate"] -= 2
            manager.record_and_evaluate(status="failed", latency_ms=None, cost=None, graph_id="g1")
            manager.close(wait=True)
        assert mock_open.call_count == 2


class TestDeliveryFailure:
    def test_delivery_failure_does_not_raise(self):
        manager = _make_manager()
        with patch("urllib.request.urlopen", side_effect=OSError("network error")):
            # Must not raise
            manager.record_and_evaluate(status="failed", latency_ms=None, cost=None, graph_id="g1")
            manager.close(wait=True)

    def test_delivery_failure_logged(self, caplog):
        import logging

        manager = _make_manager()
        with patch("urllib.request.urlopen", side_effect=OSError("network error")):
            with caplog.at_level(logging.WARNING, logger="stakeout_agent.alerts"):
                manager.record_and_evaluate(status="failed", latency_ms=None, cost=None, graph_id="g1")
                manager.close(wait=True)
        assert any("webhook delivery failed" in r.message for r in caplog.records)


class TestMultiRule:
    def test_both_rules_can_fire_independently(self):
        rules = [
            Rule(metric="error_rate", window_seconds=300, threshold=0.05),
            Rule(metric="p95_latency_ms", window_seconds=300, threshold=1000),
        ]
        manager = AlertManager(rules=rules, webhook_url="https://hooks.example.com/test", cooldown_seconds=0)
        with patch("urllib.request.urlopen") as mock_open:
            mock_open.return_value.__enter__ = lambda s: s
            mock_open.return_value.__exit__ = MagicMock(return_value=False)
            manager.record_and_evaluate(status="failed", latency_ms=5000.0, cost=None, graph_id="g1")
            manager.close(wait=True)
        assert mock_open.call_count == 2
        alerts_fired = {json.loads(c[0][0].data)["alert"] for c in mock_open.call_args_list}
        assert alerts_fired == {"error_rate", "p95_latency_ms"}

    def test_rules_with_different_cooldowns_are_independent(self):
        rules = [
            Rule(metric="error_rate", window_seconds=300, threshold=0.05),
            Rule(metric="estimated_cost_usd", window_seconds=300, threshold=1.0),
        ]
        manager = AlertManager(rules=rules, webhook_url="https://hooks.example.com/test", cooldown_seconds=9999)
        with patch("urllib.request.urlopen") as mock_open:
            mock_open.return_value.__enter__ = lambda s: s
            mock_open.return_value.__exit__ = MagicMock(return_value=False)
            # First call fires both
            manager.record_and_evaluate(status="failed", latency_ms=None, cost=5.0, graph_id="g1")
            # Second call fires neither (both in cooldown)
            manager.record_and_evaluate(status="failed", latency_ms=None, cost=5.0, graph_id="g1")
            manager.close(wait=True)
        assert mock_open.call_count == 2  # only from first call


class TestWebhookHeaders:
    def test_custom_headers_included(self):
        manager = AlertManager(
            rules=[Rule(metric="error_rate", window_seconds=300, threshold=0.05)],
            webhook_url="https://hooks.example.com/test",
            webhook_headers={"Authorization": "Bearer secret", "X-Custom": "value"},
        )
        with patch("urllib.request.urlopen") as mock_open:
            mock_open.return_value.__enter__ = lambda s: s
            mock_open.return_value.__exit__ = MagicMock(return_value=False)
            manager.record_and_evaluate(status="failed", latency_ms=None, cost=None, graph_id="g1")
            manager.close(wait=True)
        req = mock_open.call_args[0][0]
        assert req.get_header("Authorization") == "Bearer secret"
        assert req.get_header("X-custom") == "value"
        assert req.get_header("Content-type") == "application/json"


class TestMaxSamples:
    def test_sample_count_bounded_with_no_rules(self):
        manager = AlertManager(rules=[], webhook_url="http://x", max_samples=100)
        for _ in range(20_000):
            manager.record_and_evaluate(status="completed", latency_ms=10.0, cost=None, graph_id="g1")
        assert len(manager._samples) <= 100

    def test_sample_count_bounded_with_rules(self):
        manager = _make_manager(max_samples=50)
        with patch("urllib.request.urlopen") as mock_open:
            mock_open.return_value.__enter__ = lambda s: s
            mock_open.return_value.__exit__ = MagicMock(return_value=False)
            for _ in range(200):
                manager.record_and_evaluate(status="completed", latency_ms=10.0, cost=None, graph_id="g1")
            manager.close(wait=True)
        assert len(manager._samples) <= 50


class TestAsyncDelivery:
    def test_webhook_delivery_does_not_block_caller(self):
        manager = _make_manager()

        def slow_urlopen(*args, **kwargs):
            time.sleep(2)
            return MagicMock(__enter__=lambda s: s, __exit__=MagicMock(return_value=False))

        with patch("urllib.request.urlopen", side_effect=slow_urlopen) as mock_open:
            start = time.monotonic()
            manager.record_and_evaluate(status="failed", latency_ms=None, cost=None, graph_id="g1")
            elapsed = time.monotonic() - start
            assert elapsed < 0.1
            manager.close(wait=True)
        mock_open.assert_called_once()


class TestSlidingWindow:
    def test_old_samples_excluded_from_window(self):
        manager = _make_manager(rules=[Rule(metric="error_rate", window_seconds=10, threshold=0.05)])
        # Inject a sample with a very old timestamp directly
        old_sample = _RunSample(timestamp=time.time() - 100, status="failed", latency_ms=None, cost=None)
        manager._samples.append(old_sample)
        with patch("urllib.request.urlopen") as mock_open:
            mock_open.return_value.__enter__ = lambda s: s
            mock_open.return_value.__exit__ = MagicMock(return_value=False)
            # Only one completed run in the live window → 0% error rate, no alert
            manager.record_and_evaluate(status="completed", latency_ms=None, cost=None, graph_id="g1")
        mock_open.assert_not_called()
