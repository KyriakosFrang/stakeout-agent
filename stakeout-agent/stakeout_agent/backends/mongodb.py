import logging
import os
import statistics
import threading
import time
from datetime import datetime, timedelta, timezone

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


from stakeout_agent.backends.base import AbstractMonitorDB, AbstractQueryDB
from stakeout_agent.retention import RetentionPolicy

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


_RUN_PROJECTION = {
    "_id": 1,
    "graph_id": 1,
    "thread_id": 1,
    "status": 1,
    "started_at": 1,
    "ended_at": 1,
    "error": 1,
    "run_inputs": 1,
    "parent_run_id": 1,
    "prompt_version": 1,
    "total_input_tokens": 1,
    "total_output_tokens": 1,
    "estimated_cost_usd": 1,
    "total_cache_read_tokens": 1,
    "total_cache_creation_tokens": 1,
}

_EVENT_PROJECTION = {
    "_id": 0,
    "run_id": 1,
    "graph_id": 1,
    "event_type": 1,
    "node_name": 1,
    "latency_ms": 1,
    "timestamp": 1,
    "error": 1,
    "input_tokens": 1,
    "output_tokens": 1,
    "model": 1,
    "llm_output": 1,
    "cache_read_tokens": 1,
    "cache_creation_tokens": 1,
}


def _run_latency_ms(doc: dict) -> float | None:
    started = doc.get("started_at")
    ended = doc.get("ended_at")
    if started and ended:
        return (ended - started).total_seconds() * 1000
    return None


def _ser_run(doc: dict) -> dict:
    out: dict = {k: v for k, v in doc.items()}
    out["run_id"] = str(out.pop("_id"))
    for field in ("started_at", "ended_at"):
        if out.get(field) is not None:
            out[field] = out[field].isoformat()
    out["latency_ms"] = _run_latency_ms(doc)
    return out


def _ser_event(doc: dict) -> dict:
    out = {k: v for k, v in doc.items()}
    if out.get("timestamp") is not None:
        out["timestamp"] = out["timestamp"].isoformat()
    return out


def _percentile(values: list[float], p: int) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100)[p - 1]


class MongoMonitorDB(AbstractMonitorDB, AbstractQueryDB):
    def __init__(self, retention: RetentionPolicy | None = None):
        self._db = None
        self._lock = threading.Lock()
        self._retention = retention
        self._run_expires: dict[str, datetime] = {}

    @property
    def _conn(self):
        if self._db is None:
            with self._lock:
                if self._db is None:
                    self._db = _make_client()
                    if self._retention is not None:
                        self._db.runs.create_index("expires_at", expireAfterSeconds=0, sparse=True)
                        self._db.events.create_index("expires_at", expireAfterSeconds=0, sparse=True)
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

    def create_run(
        self,
        run_id: str,
        graph_id: str,
        thread_id: str,
        run_inputs: str | None = None,
        parent_run_id: str | None = None,
        prompt_version: str | None = None,
        environment: str | None = None,
    ) -> None:
        exp_at = self._retention.expires_at(graph_id=graph_id, environment=environment) if self._retention else None
        if exp_at is not None:
            self._run_expires[run_id] = exp_at

        def _op():
            doc: dict = {
                "_id": run_id,
                "graph_id": graph_id,
                "thread_id": thread_id,
                "status": "running",
                "started_at": datetime.now(timezone.utc),
                "ended_at": None,
                "error": None,
                "metadata": {},
            }
            if run_inputs is not None:
                doc["run_inputs"] = run_inputs
            if parent_run_id is not None:
                doc["parent_run_id"] = parent_run_id
            if prompt_version is not None:
                doc["prompt_version"] = prompt_version
            if exp_at is not None:
                doc["expires_at"] = exp_at
            self._conn.runs.insert_one(doc)
            _log.debug("create_run inserted run_id=%s graph_id=%s", run_id, graph_id)

        self._run_with_retry(f"create_run {run_id}", _op)

    def complete_run(
        self,
        run_id: str,
        total_input_tokens: int | None = None,
        total_output_tokens: int | None = None,
        estimated_cost_usd: float | None = None,
        total_cache_read_tokens: int | None = None,
        total_cache_creation_tokens: int | None = None,
    ) -> None:
        self._run_expires.pop(run_id, None)

        def _op():
            update: dict = {"status": "completed", "ended_at": datetime.now(timezone.utc)}
            if total_input_tokens is not None:
                update["total_input_tokens"] = total_input_tokens
            if total_output_tokens is not None:
                update["total_output_tokens"] = total_output_tokens
            if estimated_cost_usd is not None:
                update["estimated_cost_usd"] = estimated_cost_usd
            if total_cache_read_tokens is not None:
                update["total_cache_read_tokens"] = total_cache_read_tokens
            if total_cache_creation_tokens is not None:
                update["total_cache_creation_tokens"] = total_cache_creation_tokens
            result = self._conn.runs.update_one({"_id": run_id}, {"$set": update})
            if result.matched_count == 0:
                _log.warning("complete_run: no run found with id %s", run_id)
            else:
                _log.debug("complete_run run_id=%s", run_id)

        self._run_with_retry(f"complete_run {run_id}", _op)

    def fail_run(self, run_id: str, error: str) -> None:
        self._run_expires.pop(run_id, None)

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

    def prune_runs(self, older_than_days: int) -> int:
        cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
        deleted = 0

        def _op():
            nonlocal deleted
            run_ids = self._conn.runs.distinct("_id", {"started_at": {"$lt": cutoff}})
            if run_ids:
                self._conn.events.delete_many({"run_id": {"$in": run_ids}})
            result = self._conn.runs.delete_many({"started_at": {"$lt": cutoff}})
            deleted = result.deleted_count
            _log.info("prune_runs deleted %d runs older than %d days", deleted, older_than_days)

        self._run_with_retry(f"prune_runs older_than_days={older_than_days}", _op)
        return deleted

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
        cache_read_tokens: int | None = None,
        cache_creation_tokens: int | None = None,
    ) -> None:
        exp_at = self._run_expires.get(run_id)

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
            if cache_read_tokens is not None:
                doc["cache_read_tokens"] = cache_read_tokens
            if cache_creation_tokens is not None:
                doc["cache_creation_tokens"] = cache_creation_tokens
            if exp_at is not None:
                doc["expires_at"] = exp_at
            self._conn.events.insert_one(doc)
            _log.debug("insert_event event_type=%s node=%s run_id=%s", event_type, node_name, run_id)

        self._run_with_retry(f"insert_event {run_id}", _op)

    # ------------------------------------------------------------------
    # AbstractQueryDB
    # ------------------------------------------------------------------

    def query_recent_runs(self, graph_id: str | None, limit: int) -> list[dict]:
        query: dict = {}
        if graph_id:
            query["graph_id"] = graph_id
        cursor = self._conn.runs.find(query, _RUN_PROJECTION).sort("started_at", DESCENDING).limit(limit)
        return [_ser_run(doc) for doc in cursor]

    def query_run_detail(self, run_id: str) -> dict | None:
        run_doc = self._conn.runs.find_one({"_id": run_id}, _RUN_PROJECTION)
        if run_doc is None:
            return None
        events = list(self._conn.events.find({"run_id": run_id}, _EVENT_PROJECTION).sort("timestamp", 1))
        return {"run": _ser_run(run_doc), "events": [_ser_event(e) for e in events]}

    def query_failed_runs(self, graph_id: str | None, since_ts: float) -> list[dict]:
        since_dt = datetime.fromtimestamp(since_ts, tz=timezone.utc)
        query: dict = {"status": "failed", "started_at": {"$gte": since_dt}}
        if graph_id:
            query["graph_id"] = graph_id
        cursor = self._conn.runs.find(query, _RUN_PROJECTION).sort("started_at", DESCENDING)
        return [_ser_run(doc) for doc in cursor]

    def query_slow_runs(self, graph_id: str | None, threshold_ms: float, since_ts: float) -> list[dict]:
        since_dt = datetime.fromtimestamp(since_ts, tz=timezone.utc)
        query: dict = {"status": "completed", "started_at": {"$gte": since_dt}, "ended_at": {"$ne": None}}
        if graph_id:
            query["graph_id"] = graph_id
        cursor = self._conn.runs.find(query, _RUN_PROJECTION).sort("started_at", DESCENDING)
        return [_ser_run(doc) for doc in cursor if (_run_latency_ms(doc) or 0) > threshold_ms]

    def query_run_stats(self, graph_id: str | None, since_ts: float) -> dict:
        since_dt = datetime.fromtimestamp(since_ts, tz=timezone.utc)
        query: dict = {"started_at": {"$gte": since_dt}}
        if graph_id:
            query["graph_id"] = graph_id
        docs = list(self._conn.runs.find(query, _RUN_PROJECTION))
        total = len(docs)
        failed = sum(1 for d in docs if d.get("status") == "failed")
        latencies = [ms for d in docs if (ms := _run_latency_ms(d)) is not None]
        costs = [c for d in docs if (c := d.get("estimated_cost_usd")) is not None]
        return {
            "graph_id": graph_id,
            "run_count": total,
            "error_rate": (failed / total) if total else None,
            "p50_latency_ms": _percentile(latencies, 50),
            "p95_latency_ms": _percentile(latencies, 95),
            "total_cost_usd": sum(costs) if costs else None,
            "avg_cost_usd": (sum(costs) / len(costs)) if costs else None,
        }

    def query_runs_by_output(self, graph_id: str | None, text: str, limit: int) -> list[dict]:
        query: dict = {"run_inputs": {"$regex": text, "$options": "i"}}
        if graph_id:
            query["graph_id"] = graph_id
        cursor = self._conn.runs.find(query, _RUN_PROJECTION).sort("started_at", DESCENDING).limit(limit)
        return [_ser_run(doc) for doc in cursor]
