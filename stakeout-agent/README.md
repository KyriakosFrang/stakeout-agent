# stakeout-agent

Drop-in monitoring for LangGraph applications. Captures every graph run, node execution, and tool call into MongoDB or PostgreSQL with no changes to your graph code.

## Why stakeout-agent?

When building LangGraph applications, understanding how your graphs execute is critical for debugging and optimization. stakeout-agent provides:

- **Zero code changes** — just add a callback to your graph config
- **Complete visibility** — captures node starts/ends, tool calls, and errors
- **Resilient by default** — database failures are logged and never crash your application
- **MongoDB or PostgreSQL** — use whichever fits your existing infrastructure
- **Framework-agnostic core** — easily extensible to other frameworks


## Installation

```bash
# MongoDB backend (default)
pip install stakeout-agent

# PostgreSQL backend
pip install 'stakeout-agent[postgres]'
```

Requires Python 3.10+ and a running MongoDB or PostgreSQL instance.

## Quick start

### Sync (`graph.invoke`)

```python
from stakeout_agent import LangGraphMonitorCallback

monitor = LangGraphMonitorCallback(graph_id="my_graph", thread_id="thread_123")
result = graph.invoke(inputs, config={"callbacks": [monitor]})
```

### Async (`graph.ainvoke` / `graph.astream`)

```python
from stakeout_agent import AsyncLangGraphMonitorCallback

monitor = AsyncLangGraphMonitorCallback(graph_id="my_graph", thread_id="thread_123")
result = await graph.ainvoke(inputs, config={"callbacks": [monitor]})
```

## Try the example

### Run the example graph

A self-contained example graph is included to verify everything is wired up correctly.

Start MongoDB, then run:

```bash
docker compose up -d mongo
cd stakeout-agent
uv run python examples/dummy_app.py
```

It runs a three-node graph (with a tool call), then prints the `runs` and `events` documents written to MongoDB so you can confirm monitoring is working before integrating into your own application.

### Launch the dashboard

A Streamlit dashboard is included to visualise runs, node execution timelines, and tool call details.

Optionally seed demo data first, then start the dashboard:

```bash
docker compose up -d mongo
cd stakeout-agent
uv run python examples/seed_demo_data.py   # optional: load demo data
uv run --with streamlit streamlit run examples/dashboard.py
```

Open `http://localhost:8501` in your browser. The dashboard auto-refreshes every 10 seconds and shows:

- **Run History** — recent runs, status, duration, and a runs-over-time chart
- **Node Performance** — average and P95 latency per node and tool, error counts
- **Run Inspector** — full event timeline for any individual run
- **Thread Deep Dive** — multi-turn conversation view across all runs in a thread

## Configuration

| Environment variable | Default | Description |
|---|---|---|
| `STAKEOUT_BACKEND` | `mongodb` | Backend to use: `mongodb` or `postgres` |
| `MONGO_URI` | `mongodb://localhost:27017` | MongoDB connection string |
| `MONGO_DB` | `stakeout` | MongoDB database name |
| `POSTGRES_URI` | `postgresql://localhost/stakeout` | PostgreSQL connection string (also reads `DATABASE_URL`) |

### Using the PostgreSQL backend

Set `STAKEOUT_BACKEND=postgres` and provide a connection string:

```bash
export STAKEOUT_BACKEND=postgres
export POSTGRES_URI=postgresql://user:password@localhost/stakeout
```

stakeout-agent automatically creates the `runs` and `events` tables on first connection, so no migration is needed.

To start a local PostgreSQL instance for development:

```bash
docker compose up -d postgres
```

The connection string for the Docker service is `postgresql://stakeout:stakeout@localhost/stakeout`.

You can also pass a backend instance directly to skip environment-variable routing:

```python
from stakeout_agent import LangGraphMonitorCallback, PostgresMonitorDB

monitor = LangGraphMonitorCallback(
    graph_id="my_graph",
    thread_id="thread_123",
    db=PostgresMonitorDB(),
)
```

## What gets recorded

### `runs` collection

One document per graph invocation.

```json
{
  "_id": "<run_id>",
  "graph_id": "my_graph",
  "thread_id": "thread_123",
  "status": "completed",
  "started_at": "2026-04-25T10:00:00Z",
  "ended_at": "2026-04-25T10:00:05Z",
  "error": null,
  "metadata": {}
}
```

`status` is one of `running`, `completed`, or `failed`.

### `events` collection

One document per node start/end, tool call, or error within a run.

Start events:

```json
{
  "run_id": "<run_id>",
  "graph_id": "my_graph",
  "event_type": "node_start",
  "node_name": "agent",
  "timestamp": "2026-04-25T10:00:02Z",
  "payload": {"inputs": "..."},
  "error": null
}
```

End events include a `latency_ms` field measuring execution time:

```json
{
  "run_id": "<run_id>",
  "graph_id": "my_graph",
  "event_type": "node_end",
  "node_name": "agent",
  "timestamp": "2026-04-25T10:00:03Z",
  "latency_ms": 1240.5,
  "payload": {"outputs": "..."},
  "error": null
}
```

| `event_type` | When | `latency_ms` |
|---|---|---|
| `node_start` | A graph node begins execution | absent |
| `node_end` | A graph node completes | present |
| `tool_call` | A tool is invoked | absent |
| `tool_result` | A tool returns a result | present |
| `error` | A node or tool raises an exception | present |

## Error handling

All database write operations catch errors and log the failure rather than propagating the exception. A monitoring failure will never take down your application. Enable `DEBUG` logging on `stakeout_agent` to see these errors:

```python
import logging
logging.getLogger("stakeout_agent").setLevel(logging.DEBUG)
```

## Using the database backends directly

### MongoDB

```python
from stakeout_agent import MonitorDB

db = MonitorDB()

# fetch all runs for a graph
runs = list(db.runs.find({"graph_id": "my_graph"}).sort("started_at", -1))

# fetch events for a specific run
events = list(db.events.find({"run_id": "<run_id>"}).sort("timestamp", 1))
```

### PostgreSQL

```python
from stakeout_agent import PostgresMonitorDB
import psycopg2

db = PostgresMonitorDB()

# fetch all runs for a graph (use a raw psycopg2 connection for queries)
conn = psycopg2.connect("postgresql://user:password@localhost/stakeout")
with conn.cursor() as cur:
    cur.execute("SELECT * FROM runs WHERE graph_id = %s ORDER BY started_at DESC", ("my_graph",))
    runs = cur.fetchall()
```

## Package structure

```
stakeout_agent/
├── backends/
│   ├── base.py        # AbstractMonitorDB — shared interface
│   ├── postgres.py    # PostgresMonitorDB
│   └── __init__.py    # get_backend() factory
├── callback_handler/
│   ├── base.py        # _MonitorBase — framework-agnostic core logic
│   ├── langgraph.py   # LangGraphMonitorCallback, AsyncLangGraphMonitorCallback
│   └── __init__.py
└── db.py              # MonitorDB (MongoDB)
```

To add support for another LLM framework, create a file under `callback_handler/` that inherits from `_MonitorBase` and implements the target framework's callback protocol.

To add support for another database, create a class that inherits from `AbstractMonitorDB` and implement the four methods: `create_run`, `complete_run`, `fail_run`, and `insert_event`.

## Dashboard

The recorded data can power a dashboard to visualize graph runs, node execution timelines, and tool call details:

![Dashboard timeline view](https://github.com/KyriakosFrang/stakeout-agent/blob/main/stakeout-agent/public/image.png?raw=true)

## License

MIT
