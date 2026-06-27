from __future__ import annotations

import argparse
import logging
import os
import sys

_log = logging.getLogger(__name__)


def _detect_backend():
    """Auto-detect which storage backend to use from environment variables."""
    mongo_uri = os.getenv("MONGO_URI")
    pg_uri = os.getenv("POSTGRES_URI") or os.getenv("DATABASE_URL")

    if mongo_uri:
        try:
            from stakeout_agent.backends.mongodb import MongoMonitorDB

            _log.info("stakeout MCP: using MongoDB backend (%s)", mongo_uri)
            return MongoMonitorDB()
        except ImportError:
            _log.error("MONGO_URI is set but pymongo is not installed. Run: pip install 'stakeout-agent[mongodb]'")
            sys.exit(1)

    if pg_uri:
        try:
            from stakeout_agent.backends.postgres import PostgresMonitorDB

            _log.info("stakeout MCP: using Postgres backend")
            return PostgresMonitorDB()
        except ImportError:
            _log.error(
                "POSTGRES_URI / DATABASE_URL is set but psycopg2 is not installed. "
                "Run: pip install 'stakeout-agent[postgres]'"
            )
            sys.exit(1)

    print(
        "Error: no storage backend detected.\n"
        "Set MONGO_URI for MongoDB or POSTGRES_URI / DATABASE_URL for PostgreSQL.",
        file=sys.stderr,
    )
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(prog="stakeout", description="stakeout-agent CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    retention_parser = sub.add_parser("retention", help="Manage data retention")
    retention_sub = retention_parser.add_subparsers(dest="retention_command", required=True)

    apply_parser = retention_sub.add_parser(
        "apply", help="Delete runs and events whose expires_at has passed (Postgres only)"
    )
    apply_parser.add_argument("--log-level", default="WARNING", help="Logging level (default: WARNING)")

    backfill_parser = retention_sub.add_parser(
        "backfill", help="Backfill expires_at on existing rows using a default TTL"
    )
    backfill_parser.add_argument(
        "--days", type=int, required=True, help="Set expires_at = started_at/timestamp + DAYS for rows missing it"
    )
    backfill_parser.add_argument("--log-level", default="WARNING", help="Logging level (default: WARNING)")

    mcp_parser = sub.add_parser("mcp", help="Start the MCP server")
    mcp_parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="Transport to use (default: stdio)",
    )
    mcp_parser.add_argument(
        "--host", default="127.0.0.1", help="Bind host for SSE / streamable-http (default: 127.0.0.1)"
    )
    mcp_parser.add_argument("--port", type=int, default=8001, help="Bind port (default: 8001)")
    mcp_parser.add_argument("--log-level", default="WARNING", help="Logging level (default: WARNING)")

    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.WARNING))

    if args.command == "mcp":
        _run_mcp(args)
    elif args.command == "retention":
        if args.retention_command == "apply":
            _run_retention_apply(args)
        elif args.retention_command == "backfill":
            _run_retention_backfill(args)


def _run_retention_apply(args: argparse.Namespace) -> None:
    """Delete Postgres rows whose expires_at < NOW()."""
    pg_uri = os.getenv("POSTGRES_URI") or os.getenv("DATABASE_URL")
    if not pg_uri:
        print("Error: POSTGRES_URI / DATABASE_URL must be set for 'retention apply'.", file=sys.stderr)
        sys.exit(1)
    try:
        import psycopg2
    except ImportError:
        print("Error: psycopg2 is required. Run: pip install 'stakeout-agent[postgres]'", file=sys.stderr)
        sys.exit(1)

    conn = psycopg2.connect(pg_uri, connect_timeout=5)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("DELETE FROM events WHERE expires_at IS NOT NULL AND expires_at < NOW()")
        events_deleted = cur.rowcount
        cur.execute("DELETE FROM runs WHERE expires_at IS NOT NULL AND expires_at < NOW()")
        runs_deleted = cur.rowcount
    conn.close()
    print(f"retention apply: deleted {runs_deleted} run(s) and {events_deleted} event(s).")


def _run_retention_backfill(args: argparse.Namespace) -> None:
    """Backfill expires_at on existing rows that have none, using a fixed TTL in days."""
    pg_uri = os.getenv("POSTGRES_URI") or os.getenv("DATABASE_URL")
    mongo_uri = os.getenv("MONGO_URI")

    if pg_uri:
        try:
            import psycopg2
        except ImportError:
            print("Error: psycopg2 is required. Run: pip install 'stakeout-agent[postgres]'", file=sys.stderr)
            sys.exit(1)
        conn = psycopg2.connect(pg_uri, connect_timeout=5)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE runs   SET expires_at = started_at + INTERVAL '%s days' WHERE expires_at IS NULL",
                (args.days,),
            )
            runs_updated = cur.rowcount
            cur.execute(
                "UPDATE events SET expires_at = timestamp   + INTERVAL '%s days' WHERE expires_at IS NULL",
                (args.days,),
            )
            events_updated = cur.rowcount
        conn.close()
        print(f"retention backfill: updated {runs_updated} run(s) and {events_updated} event(s) (Postgres).")

    elif mongo_uri:
        try:
            from pymongo import MongoClient
        except ImportError:
            print("Error: pymongo is required. Run: pip install 'stakeout-agent[mongodb]'", file=sys.stderr)
            sys.exit(1)
        from datetime import timedelta

        db_name = os.getenv("MONGO_DB", "stakeout")
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5_000)
        db = client[db_name]

        delta = timedelta(days=args.days)

        runs_result = db.runs.update_many(
            {"expires_at": {"$exists": False}},
            [{"$set": {"expires_at": {"$add": ["$started_at", int(delta.total_seconds() * 1000)]}}}],
        )
        events_result = db.events.update_many(
            {"expires_at": {"$exists": False}},
            [{"$set": {"expires_at": {"$add": ["$timestamp", int(delta.total_seconds() * 1000)]}}}],
        )
        client.close()
        print(
            f"retention backfill: updated {runs_result.modified_count} run(s)"
            f" and {events_result.modified_count} event(s) (MongoDB)."
        )
    else:
        print(
            "Error: set MONGO_URI for MongoDB or POSTGRES_URI / DATABASE_URL for PostgreSQL.",
            file=sys.stderr,
        )
        sys.exit(1)


def _run_mcp(args: argparse.Namespace) -> None:
    try:
        from stakeout_agent.mcp.server import create_server
    except ImportError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    db = _detect_backend()
    server = create_server(db)
    token = os.getenv("STAKEOUT_MCP_TOKEN")

    if args.transport == "stdio":
        server.run(transport="stdio")
        return

    # HTTP-based transports — optionally add bearer-token auth
    if token:
        try:
            from starlette.middleware.base import BaseHTTPMiddleware
            from starlette.responses import Response

            class _BearerAuth(BaseHTTPMiddleware):
                async def dispatch(self, request, call_next):
                    auth = request.headers.get("Authorization", "")
                    if auth != f"Bearer {token}":
                        return Response("Unauthorized", status_code=401)
                    return await call_next(request)

            if args.transport == "sse":
                app = server.sse_app()
            else:
                app = server.streamable_http_app()

            app.add_middleware(_BearerAuth)

            import uvicorn

            uvicorn.run(app, host=args.host, port=args.port)
        except ImportError as exc:
            print(f"Error: {exc}. Make sure starlette and uvicorn are installed.", file=sys.stderr)
            sys.exit(1)
    else:
        server.run(transport=args.transport, host=args.host, port=args.port)
