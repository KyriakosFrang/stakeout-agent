from __future__ import annotations

import logging
import os

_log = logging.getLogger(__name__)

_MIGRATIONS_DIR = os.path.dirname(__file__)


def run_migrations(db_url: str) -> None:
    try:
        from alembic import command
        from alembic.config import Config
    except ImportError as exc:
        raise ImportError(
            "alembic is required for schema migrations. Install it with: pip install 'stakeout-agent[postgres]'"
        ) from exc

    cfg = Config()
    cfg.set_main_option("script_location", _MIGRATIONS_DIR)
    cfg.set_main_option("sqlalchemy.url", db_url)
    cfg.attributes["configure_logger"] = False
    command.upgrade(cfg, "head")
    _log.debug("schema migrations applied")
