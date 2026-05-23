from __future__ import annotations

import dataclasses
import json
import logging
import statistics
import threading
import time
import urllib.request
from collections import deque

_logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Rule:
    """A single alerting rule evaluated over a sliding window of completed runs."""

    metric: str
    window_seconds: float
    threshold: float


@dataclasses.dataclass
class _RunSample:
    timestamp: float
    status: str  # "completed" | "failed"
    latency_ms: float | None
    cost: float | None


class AlertManager:
    """Evaluates sliding-window rules after each run completes and fires webhook alerts.

    Rules are evaluated after every call to ``record_and_evaluate``. Each rule fires
    at most once per ``cooldown_seconds`` to prevent alert floods. Webhook delivery
    failures are logged but never raise.
    """

    def __init__(
        self,
        rules: list[Rule],
        webhook_url: str,
        webhook_headers: dict[str, str] | None = None,
        cooldown_seconds: float = 300,
    ) -> None:
        self._rules = list(rules)
        self._webhook_url = webhook_url
        self._webhook_headers = webhook_headers or {}
        self._cooldown_seconds = cooldown_seconds
        self._lock = threading.Lock()
        self._samples: deque[_RunSample] = deque()
        self._last_fired: dict[str, float] = {}  # metric -> last fire timestamp

    def record_and_evaluate(
        self,
        *,
        status: str,
        latency_ms: float | None,
        cost: float | None,
        graph_id: str,
    ) -> None:
        """Record a completed run and evaluate all rules against the current window."""
        now = time.time()
        sample = _RunSample(timestamp=now, status=status, latency_ms=latency_ms, cost=cost)
        with self._lock:
            self._samples.append(sample)
            self._purge(now)
            all_samples = list(self._samples)
            cooldowns = dict(self._last_fired)

        for rule in self._rules:
            cutoff = now - rule.window_seconds
            window = [s for s in all_samples if s.timestamp >= cutoff]
            value = self._compute(rule.metric, window)
            if value is None or value <= rule.threshold:
                continue
            if now - cooldowns.get(rule.metric, 0.0) < self._cooldown_seconds:
                _logger.debug("alert %s suppressed by cooldown", rule.metric)
                continue
            with self._lock:
                if now - self._last_fired.get(rule.metric, 0.0) < self._cooldown_seconds:
                    continue
                self._last_fired[rule.metric] = now
            self._fire(rule, value, graph_id, now)

    def _purge(self, now: float) -> None:
        # called with lock held; removes samples outside the largest window
        if not self._rules:
            return
        max_window = max(r.window_seconds for r in self._rules)
        cutoff = now - max_window
        while self._samples and self._samples[0].timestamp < cutoff:
            self._samples.popleft()

    @staticmethod
    def _compute(metric: str, window: list[_RunSample]) -> float | None:
        if not window:
            return None
        if metric == "error_rate":
            return sum(1 for s in window if s.status == "failed") / len(window)
        if metric in ("p95_latency_ms", "p99_latency_ms"):
            latencies = [s.latency_ms for s in window if s.latency_ms is not None]
            if not latencies:
                return None
            if len(latencies) == 1:
                return latencies[0]
            p = 95 if metric == "p95_latency_ms" else 99
            return statistics.quantiles(latencies, n=100)[p - 1]
        if metric == "estimated_cost_usd":
            costs = [s.cost for s in window if s.cost is not None]
            return sum(costs) if costs else None
        return None

    def _fire(self, rule: Rule, value: float, graph_id: str, now: float) -> None:
        payload = {
            "alert": rule.metric,
            "value": value,
            "threshold": rule.threshold,
            "window_seconds": rule.window_seconds,
            "graph_id": graph_id,
            "fired_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        }
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            self._webhook_url,
            data=body,
            headers={"Content-Type": "application/json", **self._webhook_headers},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10):
                pass
            _logger.info("alert fired metric=%s value=%s graph_id=%s", rule.metric, value, graph_id)
        except Exception as exc:
            _logger.warning("alert webhook delivery failed metric=%s: %s", rule.metric, exc)
