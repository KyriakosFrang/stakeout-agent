"""Add token and cost tracking columns to runs.

Revision ID: 0002
Revises: 0001
"""
from __future__ import annotations

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE runs ADD COLUMN IF NOT EXISTS total_input_tokens          INTEGER")
    op.execute("ALTER TABLE runs ADD COLUMN IF NOT EXISTS total_output_tokens         INTEGER")
    op.execute("ALTER TABLE runs ADD COLUMN IF NOT EXISTS estimated_cost_usd          DOUBLE PRECISION")
    op.execute("ALTER TABLE runs ADD COLUMN IF NOT EXISTS total_cache_read_tokens     INTEGER")
    op.execute("ALTER TABLE runs ADD COLUMN IF NOT EXISTS total_cache_creation_tokens INTEGER")


def downgrade() -> None:
    op.execute("ALTER TABLE runs DROP COLUMN IF EXISTS total_input_tokens")
    op.execute("ALTER TABLE runs DROP COLUMN IF EXISTS total_output_tokens")
    op.execute("ALTER TABLE runs DROP COLUMN IF EXISTS estimated_cost_usd")
    op.execute("ALTER TABLE runs DROP COLUMN IF EXISTS total_cache_read_tokens")
    op.execute("ALTER TABLE runs DROP COLUMN IF EXISTS total_cache_creation_tokens")
