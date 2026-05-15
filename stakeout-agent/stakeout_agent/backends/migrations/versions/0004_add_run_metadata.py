"""Add run_inputs, parent_run_id, and prompt_version to runs.

Revision ID: 0004
Revises: 0003
"""

from __future__ import annotations

from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE runs ADD COLUMN IF NOT EXISTS run_inputs     TEXT")
    op.execute("ALTER TABLE runs ADD COLUMN IF NOT EXISTS parent_run_id TEXT")
    op.execute("ALTER TABLE runs ADD COLUMN IF NOT EXISTS prompt_version TEXT")


def downgrade() -> None:
    op.execute("ALTER TABLE runs DROP COLUMN IF EXISTS run_inputs")
    op.execute("ALTER TABLE runs DROP COLUMN IF EXISTS parent_run_id")
    op.execute("ALTER TABLE runs DROP COLUMN IF EXISTS prompt_version")
