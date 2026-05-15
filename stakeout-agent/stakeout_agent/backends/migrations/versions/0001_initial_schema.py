"""Initial schema: runs and events tables with indexes.

Revision ID: 0001
Revises: None
"""

from __future__ import annotations

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id    TEXT PRIMARY KEY,
            graph_id  TEXT,
            thread_id TEXT,
            status    TEXT DEFAULT 'running',
            started_at TIMESTAMPTZ DEFAULT NOW(),
            ended_at   TIMESTAMPTZ,
            error      TEXT
        )
    """)
    op.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id         SERIAL PRIMARY KEY,
            run_id     TEXT,
            graph_id   TEXT,
            event_type TEXT,
            node_name  TEXT,
            latency_ms DOUBLE PRECISION,
            payload    JSONB,
            error      TEXT,
            messages   JSONB,
            timestamp  TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_runs_started_at  ON runs(started_at DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_runs_graph_id    ON runs(graph_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_runs_status      ON runs(status)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_events_run_id    ON events(run_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp DESC)")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS events")
    op.execute("DROP TABLE IF EXISTS runs")
