"""Add per-event LLM metadata columns (model, token counts, prompt/response, cache).

Revision ID: 0003
Revises: 0002
"""
from __future__ import annotations

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE events ADD COLUMN IF NOT EXISTS input_tokens          INTEGER")
    op.execute("ALTER TABLE events ADD COLUMN IF NOT EXISTS output_tokens         INTEGER")
    op.execute("ALTER TABLE events ADD COLUMN IF NOT EXISTS model                 TEXT")
    op.execute("ALTER TABLE events ADD COLUMN IF NOT EXISTS llm_input             JSONB")
    op.execute("ALTER TABLE events ADD COLUMN IF NOT EXISTS llm_output            TEXT")
    op.execute("ALTER TABLE events ADD COLUMN IF NOT EXISTS cache_read_tokens     INTEGER")
    op.execute("ALTER TABLE events ADD COLUMN IF NOT EXISTS cache_creation_tokens INTEGER")


def downgrade() -> None:
    op.execute("ALTER TABLE events DROP COLUMN IF EXISTS input_tokens")
    op.execute("ALTER TABLE events DROP COLUMN IF EXISTS output_tokens")
    op.execute("ALTER TABLE events DROP COLUMN IF EXISTS model")
    op.execute("ALTER TABLE events DROP COLUMN IF EXISTS llm_input")
    op.execute("ALTER TABLE events DROP COLUMN IF EXISTS llm_output")
    op.execute("ALTER TABLE events DROP COLUMN IF EXISTS cache_read_tokens")
    op.execute("ALTER TABLE events DROP COLUMN IF EXISTS cache_creation_tokens")
