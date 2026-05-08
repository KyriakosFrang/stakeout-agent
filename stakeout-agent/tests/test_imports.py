from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from unittest.mock import MagicMock


@contextmanager
def _hide_modules(*prefixes: str):
    """Temporarily make modules (and all their submodules) unimportable.

    Inserts None sentinels so that any `import <prefix>` raises ImportError.
    Also covers already-loaded submodules like langchain_core.callbacks so that
    re-importing a package that uses them still fails correctly.
    """
    to_hide: set[str] = set(prefixes)
    for name in list(sys.modules):
        for prefix in prefixes:
            if name == prefix or name.startswith(prefix + "."):
                to_hide.add(name)

    original = {name: sys.modules.get(name, _MISSING) for name in to_hide}
    for name in to_hide:
        sys.modules[name] = None  # type: ignore[assignment]
    try:
        yield
    finally:
        for name, val in original.items():
            if val is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = val


_MISSING = object()


def _reimport_stakeout():
    """Remove stakeout_agent from sys.modules and re-import it fresh so
    import-time conditionals re-execute against whatever sys.modules contains now."""
    for key in list(sys.modules):
        if key == "stakeout_agent" or key.startswith("stakeout_agent."):
            del sys.modules[key]
    return importlib.import_module("stakeout_agent")


class TestDefaultImports:
    def test_pricing_always_available(self):
        import stakeout_agent

        assert hasattr(stakeout_agent, "ModelPricing")
        assert hasattr(stakeout_agent, "PricingMap")


class TestMissingMongodb:
    def test_mongo_class_absent_when_pymongo_missing(self):
        with _hide_modules("pymongo", "pymongo.errors", "stakeout_agent.backends.mongodb"):
            mod = _reimport_stakeout()
        assert not hasattr(mod, "MongoMonitorDB")

    def test_other_symbols_unaffected(self):
        with _hide_modules("pymongo", "pymongo.errors", "stakeout_agent.backends.mongodb"):
            mod = _reimport_stakeout()
        assert hasattr(mod, "ModelPricing")


class TestMissingPostgres:
    def test_postgres_class_always_importable(self):
        # postgres.py uses a lazy `import psycopg2` inside _make_pg_conn so the
        # module is importable even without the extra installed; PostgresMonitorDB
        # is always present in the namespace.
        import stakeout_agent

        assert hasattr(stakeout_agent, "PostgresMonitorDB")

    def test_connection_fails_without_psycopg2(self):
        from stakeout_agent.backends.postgres import PostgresMonitorDB

        db = PostgresMonitorDB.__new__(PostgresMonitorDB)
        db._conn = None
        db._lock = __import__("threading").Lock()

        with _hide_modules("psycopg2"):
            import pytest

            with pytest.raises((ImportError, Exception)):
                from stakeout_agent.backends.postgres import _make_pg_conn

                _make_pg_conn("postgresql://localhost/test")


class TestMissingLanggraph:
    def test_langgraph_classes_absent_when_langgraph_missing(self):
        with _hide_modules(
            "langgraph",
            "langchain_core",
            "stakeout_agent.callback_handler",
            "stakeout_agent.callback_handler.langgraph",
        ):
            mod = _reimport_stakeout()
        assert not hasattr(mod, "LangGraphMonitorCallback")
        assert not hasattr(mod, "AsyncLangGraphMonitorCallback")

    def test_other_symbols_unaffected(self):
        with _hide_modules(
            "langgraph",
            "langchain_core",
            "stakeout_agent.callback_handler",
            "stakeout_agent.callback_handler.langgraph",
        ):
            mod = _reimport_stakeout()
        assert hasattr(mod, "ModelPricing")


class TestMissingCrewai:
    def test_crewai_classes_absent_when_crewai_missing(self):
        with _hide_modules("crewai", "stakeout_agent.callback_handler.crewai"):
            mod = _reimport_stakeout()
        assert not hasattr(mod, "CrewAIMonitorCallback")
        assert not hasattr(mod, "AsyncCrewAIMonitorCallback")

    def test_other_symbols_unaffected(self):
        with _hide_modules("crewai", "stakeout_agent.callback_handler.crewai"):
            mod = _reimport_stakeout()
        assert hasattr(mod, "ModelPricing")
