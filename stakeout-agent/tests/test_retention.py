from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from stakeout_agent.retention import RetentionPolicy

# ---------------------------------------------------------------------------
# RetentionPolicy.resolve
# ---------------------------------------------------------------------------


class TestRetentionPolicyResolve:
    def test_default_ttl_when_no_overrides(self):
        policy = RetentionPolicy(default_days=90)
        assert policy.resolve() == 90

    def test_default_ttl_when_no_match(self):
        policy = RetentionPolicy(default_days=90, overrides={"environment:staging": 14})
        assert policy.resolve(graph_id="other-graph", environment="prod") == 90

    def test_override_by_environment(self):
        policy = RetentionPolicy(default_days=90, overrides={"environment:dev": 7, "environment:staging": 14})
        assert policy.resolve(environment="dev") == 7
        assert policy.resolve(environment="staging") == 14

    def test_override_by_graph(self):
        policy = RetentionPolicy(default_days=90, overrides={"graph:experimental": 30})
        assert policy.resolve(graph_id="experimental") == 30

    def test_environment_takes_priority_over_graph(self):
        policy = RetentionPolicy(
            default_days=90,
            overrides={"environment:dev": 7, "graph:experimental": 30},
        )
        assert policy.resolve(graph_id="experimental", environment="dev") == 7

    def test_no_policy_leaves_expires_at_unset(self):
        policy = None
        # Simulate what backends do: no policy → no expires_at computed
        exp = policy.expires_at() if policy else None
        assert exp is None


class TestRetentionPolicyExpiresAt:
    def test_expires_at_is_in_future(self):
        policy = RetentionPolicy(default_days=90)
        before = datetime.now(timezone.utc)
        exp = policy.expires_at()
        after = datetime.now(timezone.utc)
        assert before + timedelta(days=89) < exp < after + timedelta(days=91)

    def test_expires_at_uses_resolved_days(self):
        policy = RetentionPolicy(default_days=90, overrides={"environment:dev": 7})
        exp = policy.expires_at(environment="dev")
        expected_approx = datetime.now(timezone.utc) + timedelta(days=7)
        assert abs((exp - expected_approx).total_seconds()) < 2


class TestRetentionPolicyFromEnv:
    def test_returns_none_when_env_not_set(self, monkeypatch):
        monkeypatch.delenv("STAKEOUT_RETENTION_DEFAULT_DAYS", raising=False)
        assert RetentionPolicy.from_env() is None

    def test_reads_default_days_from_env(self, monkeypatch):
        monkeypatch.setenv("STAKEOUT_RETENTION_DEFAULT_DAYS", "60")
        monkeypatch.delenv("STAKEOUT_RETENTION_DEV_DAYS", raising=False)
        monkeypatch.delenv("STAKEOUT_RETENTION_STAGING_DAYS", raising=False)
        policy = RetentionPolicy.from_env()
        assert policy is not None
        assert policy.default_days == 60
        assert policy.overrides == {}

    def test_reads_environment_overrides_from_env(self, monkeypatch):
        monkeypatch.setenv("STAKEOUT_RETENTION_DEFAULT_DAYS", "90")
        monkeypatch.setenv("STAKEOUT_RETENTION_DEV_DAYS", "7")
        monkeypatch.setenv("STAKEOUT_RETENTION_STAGING_DAYS", "14")
        policy = RetentionPolicy.from_env()
        assert policy is not None
        assert policy.overrides["environment:dev"] == 7
        assert policy.overrides["environment:staging"] == 14


# ---------------------------------------------------------------------------
# MongoDB backend: expires_at set on create_run / insert_event
# ---------------------------------------------------------------------------


def _make_mock_mongo_db():
    mock_runs = MagicMock()
    mock_events = MagicMock()
    result = MagicMock()
    result.matched_count = 1
    mock_runs.update_one.return_value = result
    mock_db = MagicMock()
    mock_db.runs = mock_runs
    mock_db.events = mock_events
    return mock_db, mock_runs, mock_events


@contextmanager
def _patched_mongo(mock_db, retention=None):
    from stakeout_agent.backends.mongodb import MongoMonitorDB

    with patch("stakeout_agent.backends.mongodb._make_client", return_value=mock_db):
        yield MongoMonitorDB(retention=retention)


class TestMongoRetention:
    def test_no_retention_omits_expires_at(self):
        mock_db, mock_runs, _ = _make_mock_mongo_db()
        with _patched_mongo(mock_db, retention=None) as monitor:
            monitor.create_run("r1", "my_graph", "t1")

        doc = mock_runs.insert_one.call_args.args[0]
        assert "expires_at" not in doc

    def test_retention_sets_expires_at_on_create_run(self):
        mock_db, mock_runs, _ = _make_mock_mongo_db()
        policy = RetentionPolicy(default_days=30)
        with _patched_mongo(mock_db, retention=policy) as monitor:
            monitor.create_run("r1", "my_graph", "t1")

        doc = mock_runs.insert_one.call_args.args[0]
        assert isinstance(doc["expires_at"], datetime)
        assert doc["expires_at"] > datetime.now(timezone.utc) + timedelta(days=29)

    def test_retention_applies_environment_override(self):
        mock_db, mock_runs, _ = _make_mock_mongo_db()
        policy = RetentionPolicy(default_days=90, overrides={"environment:dev": 7})
        with _patched_mongo(mock_db, retention=policy) as monitor:
            monitor.create_run("r1", "my_graph", "t1", environment="dev")

        doc = mock_runs.insert_one.call_args.args[0]
        expected_approx = datetime.now(timezone.utc) + timedelta(days=7)
        assert abs((doc["expires_at"] - expected_approx).total_seconds()) < 5

    def test_retention_applies_graph_override(self):
        mock_db, mock_runs, _ = _make_mock_mongo_db()
        policy = RetentionPolicy(default_days=90, overrides={"graph:my_graph": 14})
        with _patched_mongo(mock_db, retention=policy) as monitor:
            monitor.create_run("r1", "my_graph", "t1")

        doc = mock_runs.insert_one.call_args.args[0]
        expected_approx = datetime.now(timezone.utc) + timedelta(days=14)
        assert abs((doc["expires_at"] - expected_approx).total_seconds()) < 5

    def test_event_inherits_expires_at_from_run(self):
        mock_db, mock_runs, mock_events = _make_mock_mongo_db()
        policy = RetentionPolicy(default_days=30)
        with _patched_mongo(mock_db, retention=policy) as monitor:
            monitor.create_run("r1", "my_graph", "t1")
            monitor.insert_event(run_id="r1", graph_id="my_graph", event_type="node_start", node_name="n")

        run_doc = mock_runs.insert_one.call_args.args[0]
        event_doc = mock_events.insert_one.call_args.args[0]
        assert event_doc["expires_at"] == run_doc["expires_at"]

    def test_event_has_no_expires_at_without_retention(self):
        mock_db, _, mock_events = _make_mock_mongo_db()
        with _patched_mongo(mock_db, retention=None) as monitor:
            monitor.create_run("r1", "my_graph", "t1")
            monitor.insert_event(run_id="r1", graph_id="my_graph", event_type="node_start", node_name="n")

        event_doc = mock_events.insert_one.call_args.args[0]
        assert "expires_at" not in event_doc

    def test_ttl_indexes_created_when_retention_configured(self):
        mock_db, _, _ = _make_mock_mongo_db()
        policy = RetentionPolicy(default_days=30)
        with _patched_mongo(mock_db, retention=policy) as monitor:
            _ = monitor._conn  # trigger lazy init

        create_index_calls = mock_db.runs.create_index.call_args_list + mock_db.events.create_index.call_args_list
        ttl_calls = [c for c in create_index_calls if c.kwargs.get("expireAfterSeconds") == 0]
        assert len(ttl_calls) == 2

    def test_no_ttl_indexes_without_retention(self):
        mock_db, _, _ = _make_mock_mongo_db()
        with _patched_mongo(mock_db, retention=None) as monitor:
            _ = monitor._conn

        create_index_calls = mock_db.runs.create_index.call_args_list + mock_db.events.create_index.call_args_list
        ttl_calls = [c for c in create_index_calls if c.kwargs.get("expireAfterSeconds") == 0]
        assert len(ttl_calls) == 0


# ---------------------------------------------------------------------------
# Postgres backend: expires_at set on create_run / insert_event
# ---------------------------------------------------------------------------


def _make_mock_pg_conn(rowcount: int = 1):
    mock_cursor = MagicMock()
    mock_cursor.rowcount = rowcount
    mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
    mock_cursor.__exit__ = MagicMock(return_value=False)
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_conn.closed = 0
    return mock_conn, mock_cursor


@contextmanager
def _patched_postgres(mock_conn, retention=None):
    from stakeout_agent.backends.postgres import PostgresMonitorDB

    with patch("stakeout_agent.backends.postgres._make_pg_conn", return_value=mock_conn):
        yield PostgresMonitorDB(retention=retention)


class TestPostgresRetention:
    def test_no_retention_passes_null_expires_at(self):
        mock_conn, mock_cursor = _make_mock_pg_conn()
        with _patched_postgres(mock_conn, retention=None) as pg:
            pg.create_run("r1", "my_graph", "t1")

        sql, params = mock_cursor.execute.call_args.args
        assert "expires_at" in sql
        assert params[-1] is None  # expires_at is the last param

    def test_retention_sets_expires_at_on_create_run(self):
        mock_conn, mock_cursor = _make_mock_pg_conn()
        policy = RetentionPolicy(default_days=30)
        with _patched_postgres(mock_conn, retention=policy) as pg:
            pg.create_run("r1", "my_graph", "t1")

        _, params = mock_cursor.execute.call_args.args
        exp_at = params[-1]
        assert isinstance(exp_at, datetime)
        assert exp_at > datetime.now(timezone.utc) + timedelta(days=29)

    def test_retention_applies_environment_override(self):
        mock_conn, mock_cursor = _make_mock_pg_conn()
        policy = RetentionPolicy(default_days=90, overrides={"environment:dev": 7})
        with _patched_postgres(mock_conn, retention=policy) as pg:
            pg.create_run("r1", "my_graph", "t1", environment="dev")

        _, params = mock_cursor.execute.call_args.args
        exp_at = params[-1]
        expected_approx = datetime.now(timezone.utc) + timedelta(days=7)
        assert abs((exp_at - expected_approx).total_seconds()) < 5

    def test_retention_applies_graph_override(self):
        mock_conn, mock_cursor = _make_mock_pg_conn()
        policy = RetentionPolicy(default_days=90, overrides={"graph:my_graph": 14})
        with _patched_postgres(mock_conn, retention=policy) as pg:
            pg.create_run("r1", "my_graph", "t1")

        _, params = mock_cursor.execute.call_args.args
        exp_at = params[-1]
        expected_approx = datetime.now(timezone.utc) + timedelta(days=14)
        assert abs((exp_at - expected_approx).total_seconds()) < 5

    def test_event_inherits_expires_at_from_run(self):
        mock_conn, mock_cursor = _make_mock_pg_conn()
        policy = RetentionPolicy(default_days=30)
        with _patched_postgres(mock_conn, retention=policy) as pg:
            pg.create_run("r1", "my_graph", "t1")
            run_params = mock_cursor.execute.call_args.args[1]
            run_exp_at = run_params[-1]

            pg.insert_event(run_id="r1", graph_id="my_graph", event_type="node_start", node_name="n")
            event_params = mock_cursor.execute.call_args.args[1]
            event_exp_at = event_params[-1]

        assert event_exp_at == run_exp_at

    def test_event_has_null_expires_at_without_retention(self):
        mock_conn, mock_cursor = _make_mock_pg_conn()
        with _patched_postgres(mock_conn, retention=None) as pg:
            pg.create_run("r1", "my_graph", "t1")
            pg.insert_event(run_id="r1", graph_id="my_graph", event_type="node_start", node_name="n")

        event_params = mock_cursor.execute.call_args.args[1]
        assert event_params[-1] is None  # expires_at
