"""
Dummy LangGraph application to manually verify stakeout-agent monitoring.

Run MongoDB first:
    docker compose up -d mongo

Then run this script:
    uv run python examples/dummy_app.py

It will print every document written to the `runs` and `events` collections.
"""

import json
import logging
import time
from typing import TypedDict

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph

from stakeout_agent import LangGraphMonitorCallback
from stakeout_agent.db import MonitorDB

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s — %(message)s")


# ---------------------------------------------------------------------------
# A trivial tool (no LLM required)
# ---------------------------------------------------------------------------

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class State(TypedDict):
    value: int
    result: int


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def double_node(state: State) -> dict:
    return {"value": state["value"] * 2}


def multiply_node(state: State, config: RunnableConfig) -> dict:
    # Invoking a @tool with the propagated config fires tool_call / tool_result callbacks.
    product = multiply.invoke({"a": state["value"], "b": 3}, config=config)
    return {"result": product}


def summary_node(state: State) -> dict:
    print(f"\n[graph] input doubled to {state['value']}, then multiplied by 3 → {state['result']}")
    return {}


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------

def build_graph():
    g = StateGraph(State)
    g.add_node("double", double_node)
    g.add_node("multiply", multiply_node)
    g.add_node("summary", summary_node)
    g.add_edge(START, "double")
    g.add_edge("double", "multiply")
    g.add_edge("multiply", "summary")
    g.add_edge("summary", END)
    return g.compile()


# ---------------------------------------------------------------------------
# Run and inspect
# ---------------------------------------------------------------------------

def main():
    graph = build_graph()

    monitor = LangGraphMonitorCallback(graph_id="dummy_graph", thread_id="thread_001")

    print("\n--- Running graph ---")
    graph.invoke({"value": 5}, config={"callbacks": [monitor]})

    # Give a moment for any async flushing (not needed for sync, but good practice)
    time.sleep(0.1)

    print("\n--- Inspecting MongoDB ---")
    db = MonitorDB()

    run = db.runs.find_one({"graph_id": "dummy_graph"}, sort=[("started_at", -1)])
    if run is None:
        print("ERROR: no run document found — is MongoDB running?")
        return

    print(f"\nrun document:\n{json.dumps(run, default=str, indent=2)}")

    events = list(db.events.find({"run_id": run["_id"]}).sort("timestamp", 1))
    print(f"\nevents ({len(events)} total):")
    for ev in events:
        latency = f"{ev['latency_ms']} ms" if "latency_ms" in ev else "-"
        print(f"  [{ev['event_type']:12s}] node={ev['node_name']:15s} latency={latency}")


if __name__ == "__main__":
    main()
