"""Add expires_at column to runs and events for TTL-based retention.

Revision ID: 0005
Revises: 0004
"""

from __future__ import annotations

from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE runs   ADD COLUMN IF NOT EXISTS expires_at TIMESTAMPTZ")
    op.execute("ALTER TABLE events ADD COLUMN IF NOT EXISTS expires_at TIMESTAMPTZ")
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_runs_expires_at   ON runs(expires_at)   WHERE expires_at IS NOT NULL"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_events_expires_at ON events(expires_at) WHERE expires_at IS NOT NULL"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_runs_expires_at")
    op.execute("DROP INDEX IF EXISTS idx_events_expires_at")
    op.execute("ALTER TABLE runs   DROP COLUMN IF EXISTS expires_at")
    op.execute("ALTER TABLE events DROP COLUMN IF EXISTS expires_at")
