from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone


@dataclass
class RetentionPolicy:
    """TTL-based data retention policy for stakeout backends.

    Overrides are matched in priority order: environment first, then graph.
    """

    default_days: int
    overrides: dict[str, int] = field(default_factory=dict)

    def resolve(self, graph_id: str | None = None, environment: str | None = None) -> int:
        """Return retention TTL in days for the given graph/environment."""
        if environment is not None:
            env_key = f"environment:{environment}"
            if env_key in self.overrides:
                return self.overrides[env_key]
        if graph_id is not None:
            graph_key = f"graph:{graph_id}"
            if graph_key in self.overrides:
                return self.overrides[graph_key]
        return self.default_days

    def expires_at(self, graph_id: str | None = None, environment: str | None = None) -> datetime:
        """Return the absolute expiry datetime for a run starting now."""
        days = self.resolve(graph_id=graph_id, environment=environment)
        return datetime.now(timezone.utc) + timedelta(days=days)

    @classmethod
    def from_env(cls) -> RetentionPolicy | None:
        """Construct a policy from STAKEOUT_RETENTION_* env vars, or None if unconfigured."""
        default_raw = os.getenv("STAKEOUT_RETENTION_DEFAULT_DAYS")
        if default_raw is None:
            return None
        overrides: dict[str, int] = {}
        for env_name, key in (
            ("STAKEOUT_RETENTION_DEV_DAYS", "environment:dev"),
            ("STAKEOUT_RETENTION_STAGING_DAYS", "environment:staging"),
        ):
            raw = os.getenv(env_name)
            if raw is not None:
                overrides[key] = int(raw)
        return cls(default_days=int(default_raw), overrides=overrides)
