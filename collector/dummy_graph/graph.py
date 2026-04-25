"""
Dummy LangGraph graph with 3 nodes and a simulated tool call.
Used to test the monitoring collector without needing an LLM.
"""
from __future__ import annotations

import random
import time
from typing import TypedDict

from langchain_core.tools import tool
from langgraph.graph import StateGraph, END


# ── State ─────────────────────────────────────────────────────────────────────

class GraphState(TypedDict):
    query: str
    plan: str
    result: str
    summary: str


# ── Simulated tools ───────────────────────────────────────────────────────────

@tool
def search_database(query: str) -> str:
    """Search the database for relevant records."""
    time.sleep(random.uniform(0.05, 0.2))
    if random.random() < 0.15:
        raise ValueError(f"Database connection timeout for query: {query}")
    return f"Found 3 records matching '{query}': [Record A, Record B, Record C]"


@tool
def calculate_metric(value: str) -> str:
    """Calculate a business metric from input value."""
    time.sleep(random.uniform(0.02, 0.1))
    return f"Metric calculated: {hash(value) % 1000 / 10:.1f}%"


# ── Nodes ─────────────────────────────────────────────────────────────────────

def planner_node(state: GraphState) -> GraphState:
    """Plans the execution based on the query."""
    time.sleep(random.uniform(0.1, 0.3))
    state["plan"] = f"Plan: retrieve data for '{state['query']}', then calculate metrics"
    return state


def executor_node(state: GraphState) -> GraphState:
    """Executes the plan by calling tools."""
    time.sleep(random.uniform(0.05, 0.15))

    # Simulate tool calls directly (without LLM)
    try:
        db_result = search_database.invoke({"query": state["query"]})
        metric_result = calculate_metric.invoke({"value": db_result})
        state["result"] = f"{db_result} | {metric_result}"
    except Exception as e:
        state["result"] = f"Execution failed: {str(e)}"

    return state


def summarizer_node(state: GraphState) -> GraphState:
    """Summarizes the results."""
    time.sleep(random.uniform(0.05, 0.2))

    if random.random() < 0.1:
        raise RuntimeError("Summarizer LLM call failed: rate limit exceeded")

    state["summary"] = (
        f"Query '{state['query']}' processed successfully. "
        f"Result: {state['result'][:100]}"
    )
    return state


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(GraphState)

    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("summarizer", summarizer_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "summarizer")
    graph.add_edge("summarizer", END)

    return graph.compile()
