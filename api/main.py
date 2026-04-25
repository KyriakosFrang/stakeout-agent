import os
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient, DESCENDING

app = FastAPI(title="LangGraph Monitor API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── DB connection ─────────────────────────────────────────────────────────────

def get_db():
    uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    db_name = os.getenv("MONGO_DB", "stakeout")
    client = MongoClient(uri)
    return client[db_name]


def serialize(doc: dict) -> dict:
    """Make MongoDB doc JSON-serializable."""
    doc["id"] = str(doc.pop("_id", ""))
    for k, v in doc.items():
        if isinstance(v, datetime):
            doc[k] = v.isoformat()
    return doc


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/runs")
def list_runs(
    limit: int = Query(default=50, le=200),
    graph_id: str | None = Query(default=None),
    status: str | None = Query(default=None),
):
    db = get_db()
    query: dict[str, Any] = {}
    if graph_id:
        query["graph_id"] = graph_id
    if status:
        query["status"] = status

    runs = list(
        db.runs.find(query)
        .sort("started_at", DESCENDING)
        .limit(limit)
    )
    return [serialize(r) for r in runs]


@app.get("/runs/{run_id}")
def get_run(run_id: str):
    db = get_db()
    run = db.runs.find_one({"_id": run_id})
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return serialize(run)


@app.get("/runs/{run_id}/events")
def get_run_events(run_id: str):
    db = get_db()
    run = db.runs.find_one({"_id": run_id})
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    events = list(db.events.find({"run_id": run_id}).sort("timestamp", 1))
    return [serialize(e) for e in events]


@app.get("/stats")
def get_stats():
    db = get_db()

    total = db.runs.count_documents({})
    completed = db.runs.count_documents({"status": "completed"})
    failed = db.runs.count_documents({"status": "failed"})
    running = db.runs.count_documents({"status": "running"})

    # Average latency per graph from events
    pipeline = [
        {"$match": {"event_type": "node_end", "latency_ms": {"$ne": None}}},
        {"$group": {"_id": "$graph_id", "avg_latency_ms": {"$avg": "$latency_ms"}}},
    ]
    latency_by_graph = {
        doc["_id"]: round(doc["avg_latency_ms"], 2)
        for doc in db.events.aggregate(pipeline)
    }

    # Error count by graph
    error_pipeline = [
        {"$match": {"event_type": "error"}},
        {"$group": {"_id": "$graph_id", "error_count": {"$sum": 1}}},
    ]
    errors_by_graph = {
        doc["_id"]: doc["error_count"]
        for doc in db.events.aggregate(error_pipeline)
    }

    # Graph IDs
    graph_ids = db.runs.distinct("graph_id")

    return {
        "total_runs": total,
        "completed": completed,
        "failed": failed,
        "running": running,
        "success_rate": round(completed / total * 100, 1) if total else 0,
        "graph_ids": graph_ids,
        "avg_latency_ms_by_graph": latency_by_graph,
        "errors_by_graph": errors_by_graph,
    }


@app.get("/graphs")
def list_graphs():
    db = get_db()
    graph_ids = db.runs.distinct("graph_id")
    results = []
    for gid in graph_ids:
        total = db.runs.count_documents({"graph_id": gid})
        failed = db.runs.count_documents({"graph_id": gid, "status": "failed"})
        results.append({
            "graph_id": gid,
            "total_runs": total,
            "failed_runs": failed,
            "success_rate": round((total - failed) / total * 100, 1) if total else 0,
        })
    return results
