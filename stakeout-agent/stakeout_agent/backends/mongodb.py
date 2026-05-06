import logging
import os
import threading
from datetime import datetime, timezone

from pymongo import DESCENDING, MongoClient
from pymongo.collection import Collection
from pymongo.errors import PyMongoError

from stakeout_agent.backends.base import AbstractMonitorDB

_log = logging.getLogger(__name__)


def _make_client():
    uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    db_name = os.getenv("MONGO_DB", "stakeout")
    client = MongoClient(uri)
    db = client[db_name]
    db.runs.create_index([("started_at", DESCENDING)])
    db.runs.create_index("graph_id")
    db.runs.create_index("status")
    db.events.create_index("run_id")
    db.events.create_index([("timestamp", DESCENDING)])
    _log.debug("MonitorDB connected uri=%s db=%s", uri, db_name)
    return db


class MongoMonitorDB(AbstractMonitorDB):
    def __init__(self):
        self._db = None
        self._lock = threading.Lock()

    @property
    def _conn(self):
        if self._db is None:
            with self._lock:
                if self._db is None:  # double-checked locking
                    self._db = _make_client()
        return self._db

    @property
    def runs(self) -> Collection:
        return self._conn.runs

    @property
    def events(self) -> Collection:
        return self._conn.events

    def create_run(self, run_id: str, graph_id: str, thread_id: str) -> None:
        runs = self.runs
        try:
            runs.insert_one(
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
        except PyMongoError as exc:
            _log.error("create_run %s failed: %s", run_id, exc)
            return
        _log.debug("create_run inserted run_id=%s graph_id=%s", run_id, graph_id)

    def complete_run(
        self,
        run_id: str,
        total_input_tokens: int | None = None,
        total_output_tokens: int | None = None,
        estimated_cost_usd: float | None = None,
    ) -> None:
        runs = self.runs
        update: dict = {"status": "completed", "ended_at": datetime.now(timezone.utc)}
        if total_input_tokens is not None:
            update["total_input_tokens"] = total_input_tokens
        if total_output_tokens is not None:
            update["total_output_tokens"] = total_output_tokens
        if estimated_cost_usd is not None:
            update["estimated_cost_usd"] = estimated_cost_usd
        try:
            result = runs.update_one({"_id": run_id}, {"$set": update})
        except PyMongoError as exc:
            _log.error("complete_run %s failed: %s", run_id, exc)
            return
        if result.matched_count == 0:
            _log.warning("complete_run: no run found with id %s", run_id)
        else:
            _log.debug("complete_run run_id=%s", run_id)

    def fail_run(self, run_id: str, error: str) -> None:
        runs = self.runs
        try:
            result = runs.update_one(
                {"_id": run_id}, {"$set": {"status": "failed", "ended_at": datetime.now(timezone.utc), "error": error}}
            )
        except PyMongoError as exc:
            _log.error("fail_run %s failed: %s", run_id, exc)
            return
        if result.matched_count == 0:
            _log.warning("fail_run: no run found with id %s", run_id)
        else:
            _log.debug("fail_run run_id=%s", run_id)

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
    ) -> None:
        events = self.events
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
        try:
            events.insert_one(doc)
        except PyMongoError as exc:
            _log.error("insert_event for run %s failed: %s", run_id, exc)
            return
        _log.debug("insert_event event_type=%s node=%s run_id=%s", event_type, node_name, run_id)
