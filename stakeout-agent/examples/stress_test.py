"""
LangGraph Stabilization & Stress Test

Verifies LangGraphMonitorCallback works correctly end-to-end against a real
MongoDB instance under load. No events, runs, or token data should be dropped.

Usage:
    docker compose up -d mongo
    cd stakeout-agent
    uv run python examples/stress_test.py
"""
from __future__ import annotations

import asyncio
import os
import random
import sys
import time
from pathlib import Path

from langchain_core.messages import HumanMessage
from pymongo import MongoClient

sys.path.insert(0, str(Path(__file__).parent))

from langgraph_seed import _CS_INPUTS, PRICING, build_customer_support_graph

from stakeout_agent import AsyncLangGraphMonitorCallback, LangGraphMonitorCallback

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = "stakeout_stress"
GRAPH_ID = "customer_support_agent"
THREADS = [f"thread_{i:03d}" for i in range(1, 16)]
CS_NODES = ["classify_intent", "retrieve_context", "draft_reply", "quality_check"]
FAILURE_RATE = 0.15


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_inputs(inject_error: str = "") -> dict:
    return {
        "messages": [HumanMessage(content=random.choice(_CS_INPUTS))],
        "context": "",
        "intent": "",
        "inject_error": inject_error,
    }


def _maybe_fail() -> str:
    return random.choice(CS_NODES) if random.random() < FAILURE_RATE else ""


# ---------------------------------------------------------------------------
# Assertion engine
# ---------------------------------------------------------------------------

_WIDTH = 60


def assert_batch(db, expected_total: int, monitors: list, label: str) -> None:
    failures: list[str] = []

    print(f"\n{'─' * _WIDTH}")
    print(f"  {label}")
    print(f"{'─' * _WIDTH}")

    def check(name: str, passed: bool, detail: str = "") -> None:
        status = "PASS" if passed else "FAIL"
        suffix = f" — {detail}" if detail else ""
        print(f"  [{status}] {name}{suffix}")
        if not passed:
            failures.append(f"{name}: {detail}")

    # 1. Run count
    actual_total = db.runs.count_documents({})
    check(
        "Run count",
        actual_total == expected_total,
        f"expected {expected_total}, got {actual_total}",
    )

    # 2. No orphaned runs
    orphaned = db.runs.count_documents({"status": "running"})
    check("No orphaned runs", orphaned == 0, f"{orphaned} stuck in 'running'")

    # 3. Event completeness — every completed run has ≥1 node_start and ≥1 node_end
    completed_runs = list(db.runs.find({"status": "completed"}, {"_id": 1}))
    incomplete = []
    for run_doc in completed_runs:
        rid = run_doc["_id"]
        ns = db.events.count_documents({"run_id": rid, "event_type": "node_start"})
        ne = db.events.count_documents({"run_id": rid, "event_type": "node_end"})
        if ns < 1 or ne < 1:
            incomplete.append(rid)
    check(
        "Event completeness",
        len(incomplete) == 0,
        f"{len(incomplete)} completed runs missing node events",
    )

    # 4. Token tracking — all completed runs must have input tokens > 0
    no_tokens = db.runs.count_documents(
        {"status": "completed", "$or": [{"total_input_tokens": {"$lte": 0}}, {"total_input_tokens": None}]}
    )
    check("Token tracking", no_tokens == 0, f"{no_tokens} completed runs with zero/null input tokens")

    # 5. Cost tracking — all completed runs must have estimated_cost_usd > 0
    no_cost = db.runs.count_documents(
        {"status": "completed", "$or": [{"estimated_cost_usd": {"$lte": 0}}, {"estimated_cost_usd": None}]}
    )
    check("Cost tracking", no_cost == 0, f"{no_cost} completed runs with zero/null cost")

    # 6. No silent drops
    if monitors:
        total_dropped = sum(m.dropped_events for m in monitors)
        check("No silent drops", total_dropped == 0, f"{total_dropped} dropped DB writes across {len(monitors)} monitors")

    # 7. Failure handling — no run stuck in an unexpected state
    bad_status = db.runs.count_documents({"status": {"$nin": ["completed", "failed"]}})
    check("Failure handling", bad_status == 0, f"{bad_status} runs with unexpected status")

    print(f"{'─' * _WIDTH}")
    if failures:
        print(f"  {len(failures)} assertion(s) FAILED")
        for f in failures:
            print(f"    ✗ {f}")
        print()
        raise SystemExit(1)
    else:
        print(f"  All assertions passed\n")


# ---------------------------------------------------------------------------
# Batch 1 — 200 sequential sync runs
# ---------------------------------------------------------------------------


def run_sequential_batch(graph, n: int = 200) -> tuple[int, int, list]:
    monitors: list = []
    completed = failed = 0
    print(f"\nBatch 1: {n} sequential sync runs  (failure_rate={FAILURE_RATE:.0%})")
    for i in range(n):
        inject = _maybe_fail()
        monitor = LangGraphMonitorCallback(
            graph_id=GRAPH_ID,
            thread_id=random.choice(THREADS),
            pricing=PRICING,
        )
        try:
            graph.invoke(_make_inputs(inject), config={"callbacks": [monitor]})
            completed += 1
        except Exception:
            failed += 1
        monitors.append(monitor)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n}  completed={completed}  failed={failed}")
    return completed, failed, monitors


# ---------------------------------------------------------------------------
# Batch 2 — 50 concurrent async runs
# ---------------------------------------------------------------------------


async def _single_async_run(graph, inject: str) -> tuple[str, AsyncLangGraphMonitorCallback]:
    monitor = AsyncLangGraphMonitorCallback(
        graph_id=GRAPH_ID,
        thread_id=random.choice(THREADS),
        pricing=PRICING,
    )
    try:
        await graph.ainvoke(_make_inputs(inject), config={"callbacks": [monitor]})
        return "completed", monitor
    except Exception:
        return "failed", monitor


async def _async_batch(graph, n: int) -> tuple[int, int, list]:
    tasks = [_single_async_run(graph, _maybe_fail()) for _ in range(n)]
    results = await asyncio.gather(*tasks)
    completed = sum(1 for s, _ in results if s == "completed")
    failed = sum(1 for s, _ in results if s == "failed")
    monitors = [m for _, m in results]
    return completed, failed, monitors


def run_async_batch(graph, n: int = 50) -> tuple[int, int, list]:
    print(f"\nBatch 2: {n} concurrent async runs  (failure_rate={FAILURE_RATE:.0%})")
    completed, failed, monitors = asyncio.run(_async_batch(graph, n))
    print(f"  {n}/{n}  completed={completed}  failed={failed}")
    return completed, failed, monitors


# ---------------------------------------------------------------------------
# Batch 3 — Repeated runs on same thread_id (thread isolation)
# ---------------------------------------------------------------------------


def run_thread_repeat_batch(graph, db, n: int = 20) -> list:
    thread_id = "thread_stress_repeat"
    print(f"\nBatch 3: {n} sequential runs on thread_id='{thread_id}'")
    before_count = db.runs.count_documents({"thread_id": thread_id})
    monitors: list = []

    for _ in range(n):
        monitor = LangGraphMonitorCallback(
            graph_id=GRAPH_ID,
            thread_id=thread_id,
            pricing=PRICING,
        )
        try:
            graph.invoke(_make_inputs(), config={"callbacks": [monitor]})
        except Exception:
            pass
        monitors.append(monitor)

    after_count = db.runs.count_documents({"thread_id": thread_id})
    new_count = after_count - before_count
    if new_count != n:
        print(f"  [FAIL] Expected {n} new runs for thread, got {new_count}")
        raise SystemExit(1)

    # Verify each completed run on this thread has ≥4 events (4 node_start + 4 node_end)
    thread_runs = list(
        db.runs.find({"thread_id": thread_id, "status": "completed"}, {"_id": 1}).sort("_id", -1).limit(n)
    )
    for run_doc in thread_runs:
        rid = run_doc["_id"]
        event_count = db.events.count_documents({"run_id": rid})
        if event_count < 4:
            print(f"  [FAIL] Run {rid} on repeat thread has only {event_count} events (expected ≥4)")
            raise SystemExit(1)

    print(f"  {n} runs on '{thread_id}' — each run isolated, event counts OK")
    return monitors


# ---------------------------------------------------------------------------
# Batch 4 — Concurrency isolation check (10 simultaneous async)
# ---------------------------------------------------------------------------


async def _isolation_check(graph, db) -> None:
    n = 10
    run_ids_before = {r["_id"] for r in db.runs.find({}, {"_id": 1})}

    tasks = [_single_async_run(graph, "") for _ in range(n)]
    await asyncio.gather(*tasks)

    run_ids_after = {r["_id"] for r in db.runs.find({}, {"_id": 1})}
    new_run_ids = run_ids_after - run_ids_before

    if len(new_run_ids) != n:
        raise AssertionError(f"Expected {n} new runs in isolation check, got {len(new_run_ids)}")

    # Every event's run_id must belong to a known run (no orphaned events)
    event_run_ids = set(db.events.distinct("run_id"))
    orphaned = event_run_ids - run_ids_after
    if orphaned:
        raise AssertionError(f"Orphaned event run_ids (no matching run doc): {orphaned}")

    # Each new run must have at least one event written to it
    for rid in new_run_ids:
        count = db.events.count_documents({"run_id": rid})
        if count < 1:
            raise AssertionError(f"Run {rid} from isolation batch has 0 events")

    # Completed runs must have node_start and node_end events
    completed_new = list(db.runs.find({"_id": {"$in": list(new_run_ids)}, "status": "completed"}, {"_id": 1}))
    for run_doc in completed_new:
        rid = run_doc["_id"]
        ns = db.events.count_documents({"run_id": rid, "event_type": "node_start"})
        ne = db.events.count_documents({"run_id": rid, "event_type": "node_end"})
        if ns < 1 or ne < 1:
            raise AssertionError(f"Run {rid} missing node events (node_start={ns}, node_end={ne})")


def run_isolation_check(graph, db) -> None:
    print(f"\nBatch 4: 10 simultaneous async runs — concurrency isolation check")
    try:
        asyncio.run(_isolation_check(graph, db))
        print("  No cross-contamination detected")
    except AssertionError as exc:
        print(f"  [FAIL] {exc}")
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    os.environ.setdefault("MONGO_URI", MONGO_URI)
    os.environ.setdefault("MONGO_DB", DB_NAME)

    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    db.runs.drop()
    db.events.drop()
    print(f"Cleared '{DB_NAME}.runs' and '{DB_NAME}.events'")

    graph = build_customer_support_graph()
    t_start = time.monotonic()

    # Batch 1: 200 sequential sync runs
    _, _, m1 = run_sequential_batch(graph, n=200)
    assert_batch(db, expected_total=200, monitors=m1, label="Batch 1 — sequential sync (200 runs)")

    # Batch 2: 50 concurrent async runs
    _, _, m2 = run_async_batch(graph, n=50)
    assert_batch(db, expected_total=250, monitors=m2, label="Batch 2 — concurrent async (50 runs)")

    # Batch 3: 20 repeated runs on same thread_id
    m3 = run_thread_repeat_batch(graph, db, n=20)
    assert_batch(db, expected_total=270, monitors=m3, label="Batch 3 — thread repeat (20 runs)")

    # Batch 4: 10 simultaneous async — concurrency isolation
    run_isolation_check(graph, db)
    assert_batch(db, expected_total=280, monitors=[], label="Batch 4 — concurrency isolation (10 runs)")

    elapsed = time.monotonic() - t_start
    total_runs = db.runs.count_documents({})
    total_events = db.events.count_documents({})
    completed = db.runs.count_documents({"status": "completed"})
    failed = db.runs.count_documents({"status": "failed"})

    print(f"{'═' * _WIDTH}")
    print(f"  All checks passed in {elapsed:.1f}s")
    print(f"  Runs: {total_runs} total  ({completed} completed, {failed} failed)")
    print(f"  Events: {total_events}")
    print(f"{'═' * _WIDTH}")


if __name__ == "__main__":
    main()
