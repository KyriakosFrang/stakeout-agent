import os
from datetime import datetime, timezone

from pymongo import DESCENDING, MongoClient
from pymongo.collection import Collection


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
    return db


class MonitorDB:
    def __init__(self):
        self._db = None

    @property
    def _conn(self):
        if self._db is None:
            self._db = _make_client()
        return self._db

    @property
    def runs(self) -> Collection:
        return self._conn.runs

    @property
    def events(self) -> Collection:
        return self._conn.events

    def create_run(self, run_id: str, graph_id: str, thread_id: str) -> None:
        self.runs.insert_one(
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

    def complete_run(self, run_id: str) -> None:
        self.runs.update_one({"_id": run_id}, {"$set": {"status": "completed", "ended_at": datetime.now(timezone.utc)}})

    def fail_run(self, run_id: str, error: str) -> None:
        self.runs.update_one(
            {"_id": run_id}, {"$set": {"status": "failed", "ended_at": datetime.now(timezone.utc), "error": error}}
        )

    def insert_event(
        self,
        run_id: str,
        graph_id: str,
        event_type: str,
        node_name: str,
        latency_ms: float | None = None,
        payload: dict | None = None,
        error: str | None = None,
    ) -> None:
        self.events.insert_one(
            {
                "run_id": run_id,
                "graph_id": graph_id,
                "event_type": event_type,
                "node_name": node_name,
                "timestamp": datetime.now(timezone.utc),
                "latency_ms": latency_ms,
                "payload": payload or {},
                "error": error,
            }
        )
