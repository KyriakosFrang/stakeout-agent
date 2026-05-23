from __future__ import annotations

import time

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:
    raise ImportError(
        "mcp is required for the MCP server. Install it with: pip install 'stakeout-agent[mcp]'"
    ) from exc

from stakeout_agent.backends.base import AbstractQueryDB


def create_server(db: AbstractQueryDB) -> FastMCP:
    """Build and return a FastMCP server wired to *db*."""
    server = FastMCP("stakeout")

    @server.tool()
    def get_recent_runs(graph_id: str = "", limit: int = 10) -> list[dict]:
        """Return the N most recent runs for a graph.

        Args:
            graph_id: Filter by graph identifier. Pass empty string to return runs across all graphs.
            limit: Maximum number of runs to return (1-100).
        """
        return db.query_recent_runs(graph_id or None, max(1, min(limit, 100)))

    @server.tool()
    def get_run_detail(run_id: str) -> dict:
        """Return the full event trace for a specific run.

        Args:
            run_id: The unique run identifier.
        """
        result = db.query_run_detail(run_id)
        if result is None:
            return {"error": f"Run {run_id!r} not found"}
        return result

    @server.tool()
    def get_failed_runs(graph_id: str = "", last_n_hours: float = 24.0) -> list[dict]:
        """Return failed runs within a time window, with error messages.

        Args:
            graph_id: Filter by graph identifier. Pass empty string for all graphs.
            last_n_hours: How many hours back to search (default 24).
        """
        since_ts = time.time() - last_n_hours * 3600
        return db.query_failed_runs(graph_id or None, since_ts)

    @server.tool()
    def get_slow_runs(threshold_ms: float, graph_id: str = "", last_n_hours: float = 24.0) -> list[dict]:
        """Return runs that exceeded a latency threshold.

        Args:
            threshold_ms: Latency threshold in milliseconds.
            graph_id: Filter by graph identifier. Pass empty string for all graphs.
            last_n_hours: How many hours back to search (default 24).
        """
        since_ts = time.time() - last_n_hours * 3600
        return db.query_slow_runs(graph_id or None, threshold_ms, since_ts)

    @server.tool()
    def get_run_stats(graph_id: str = "", last_n_days: float = 7.0) -> dict:
        """Return aggregate stats for a graph: error rate, p50/p95 latency, total cost.

        Args:
            graph_id: Filter by graph identifier. Pass empty string for all graphs.
            last_n_days: How many days back to include (default 7).
        """
        since_ts = time.time() - last_n_days * 86400
        return db.query_run_stats(graph_id or None, since_ts)

    @server.tool()
    def search_runs_by_output(query: str, graph_id: str = "", limit: int = 20) -> list[dict]:
        """Full-text search across captured run inputs/outputs.

        Args:
            query: Search term (case-insensitive substring match).
            graph_id: Filter by graph identifier. Pass empty string for all graphs.
            limit: Maximum number of results (1-100).
        """
        return db.query_runs_by_output(graph_id or None, query, max(1, min(limit, 100)))

    @server.resource("stakeout://runs")
    def list_runs_resource() -> str:
        """List of the 20 most recent runs across all graphs."""
        import json

        runs = db.query_recent_runs(None, 20)
        return json.dumps(runs, indent=2)

    @server.resource("stakeout://runs/{run_id}")
    def run_detail_resource(run_id: str) -> str:
        """Full trace for a specific run."""
        import json

        result = db.query_run_detail(run_id)
        if result is None:
            return json.dumps({"error": f"Run {run_id!r} not found"})
        return json.dumps(result, indent=2)

    @server.resource("stakeout://graphs/{graph_id}/stats")
    def graph_stats_resource(graph_id: str) -> str:
        """Aggregate stats for a graph over the last 7 days."""
        import json

        since_ts = time.time() - 7 * 86400
        stats = db.query_run_stats(graph_id, since_ts)
        return json.dumps(stats, indent=2)

    return server
