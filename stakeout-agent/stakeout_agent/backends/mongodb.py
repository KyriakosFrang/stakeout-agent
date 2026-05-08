import logging
import os
import threading
import time
from datetime import datetime, timezone

try:
    from pymongo import DESCENDING, MongoClient
    from pymongo.collection import Collection
    from pymongo.errors import ConnectionFailure, PyMongoError
except ImportError:
    DESCENDING = None
    MongoClient = None
    Collection = None

    class ConnectionFailure(Exception):  # type: ignore[no-redef]
        pass

    class PyMongoError(Exception):  # type: ignore[no-redef]
        pass


from stakeout_agent.backends.base import AbstractMonitorDB

_log = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE = 0.5  # seconds; doubles each attempt


def _make_client():
    if MongoClient is None:
        raise ImportError(
            "pymongo is required for the MongoDB backend. Install it with: pip install 'stakeout-agent[mongodb]'"
        )
    uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    db_name = os.getenv("MONGO_DB", "stakeout")
    client = MongoClient(
        uri,
        retryWrites=True,
        retryReads=True,
        serverSelectionTimeoutMS=5_000,
        connectTimeoutMS=5_000,
        socketTimeoutMS=20_000,
        maxPoolSize=10,
        minPoolSize=1,
    )
    db = client[db_name]
    db.runs.create_index([("started_at", DESCENDING)])
    db.runs.create_index("graph_id")
    db.runs.create_index("status")
    db.events.create_index("run_id")
    db.events.create_index([("timestamp", DESCENDING)])
    _log.debug("MongoMonitorDB connected uri=%s db=%s", uri, db_name)
    return db


class MongoMonitorDB(AbstractMonitorDB):
    def __init__(self):
        self._db = None
        self._lock = threading.Lock()

    @property
    def _conn(self):
        if self._db is None:
            with self._lock:
                if self._db is None:
                    self._db = _make_client()
        return self._db

    @property
    def runs(self) -> Collection:
        return self._conn.runs

    @property
    def events(self) -> Collection:
        return self._conn.events

    def _reset_conn(self) -> None:
        with self._lock:
            self._db = None

    def _run_with_retry(self, op_name: str, fn) -> None:
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                fn()
                return
            except ConnectionFailure as exc:
                self._reset_conn()
                if attempt < _MAX_RETRIES:
                    delay = _RETRY_BACKOFF_BASE * (2 ** (attempt - 1))
                    _log.warning(
                        "%s attempt %d/%d failed: %s — retrying in %.1fs",
                        op_name,
                        attempt,
                        _MAX_RETRIES,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
                else:
                    _log.error("%s failed after %d attempts: %s", op_name, _MAX_RETRIES, exc)
            except PyMongoError as exc:
                _log.error("%s failed: %s", op_name, exc)
                return

    def create_run(self, run_id: str, graph_id: str, thread_id: str) -> None:
        def _op():
            self._conn.runs.insert_one(
                {
                    "_id": run_id,
                    "graph_id": graph_id,
                    "thread_id": thread_id,
                    "status": "running",
                    "started_at": datetime.now(timezone.utc),
                    "ended_at": None,
                    "error": None,
                    "metadata": {},
                }
            )
            _log.debug("create_run inserted run_id=%s graph_id=%s", run_id, graph_id)

        self._run_with_retry(f"create_run {run_id}", _op)

    def complete_run(
        self,
        run_id: str,
        total_input_tokens: int | None = None,
        total_output_tokens: int | None = None,
        estimated_cost_usd: float | None = None,
    ) -> None:
        def _op():
            update: dict = {"status": "completed", "ended_at": datetime.now(timezone.utc)}
            if total_input_tokens is not None:
                update["total_input_tokens"] = total_input_tokens
            if total_output_tokens is not None:
                update["total_output_tokens"] = total_output_tokens
            if estimated_cost_usd is not None:
                update["estimated_cost_usd"] = estimated_cost_usd
            result = self._conn.runs.update_one({"_id": run_id}, {"$set": update})
            if result.matched_count == 0:
                _log.warning("complete_run: no run found with id %s", run_id)
            else:
                _log.debug("complete_run run_id=%s", run_id)

        self._run_with_retry(f"complete_run {run_id}", _op)

    def fail_run(self, run_id: str, error: str) -> None:
        def _op():
            result = self._conn.runs.update_one(
                {"_id": run_id},
                {"$set": {"status": "failed", "ended_at": datetime.now(timezone.utc), "error": error}},
            )
            if result.matched_count == 0:
                _log.warning("fail_run: no run found with id %s", run_id)
            else:
                _log.debug("fail_run run_id=%s", run_id)

        self._run_with_retry(f"fail_run {run_id}", _op)

    def insert_event(
        self,
        run_id: str,
        graph_id: str,
        event_type: str,
        node_name: str,
        latency_ms: float | None = None,
        payload: dict | None = None,
        error: str | None = None,
        messages: list[dict] | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        model: str | None = None,
        llm_input: list[dict] | None = None,
        llm_output: str | None = None,
    ) -> None:
        def _op():
            doc: dict = {
                "run_id": run_id,
                "graph_id": graph_id,
                "event_type": event_type,
                "node_name": node_name,
                "timestamp": datetime.now(timezone.utc),
                "payload": payload or {},
                "error": error,
            }
            if latency_ms is not None:
                doc["latency_ms"] = latency_ms
            if messages is not None:
                doc["messages"] = messages
            if input_tokens is not None:
                doc["input_tokens"] = input_tokens
            if output_tokens is not None:
                doc["output_tokens"] = output_tokens
            if model is not None:
                doc["model"] = model
            if llm_input is not None:
                doc["llm_input"] = llm_input
            if llm_output is not None:
                doc["llm_output"] = llm_output
            self._conn.events.insert_one(doc)
            _log.debug("insert_event event_type=%s node=%s run_id=%s", event_type, node_name, run_id)

        self._run_with_retry(f"insert_event {run_id}", _op)
