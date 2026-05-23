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
