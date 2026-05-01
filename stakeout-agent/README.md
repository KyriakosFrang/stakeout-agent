# stakeout-agent

<h1 align="center">stakeout-agent</h1>

<p align="center">
  <strong>Drop-in observability for LangGraph and CrewAI.</strong>
</p>

<p align="center">
   One callback. Every run, node, and tool call — captured automatically into MongoDB or PostgreSQL. No changes to your agent code.
</p>

<p align="center">
  <a href="https://pypi.org/project/stakeout-agent/">
    <img src="https://img.shields.io/pypi/v/stakeout-agent" alt="PyPI">
  </a>
  <a href="https://pypi.org/project/stakeout-agent/">
    <img src="https://img.shields.io/pypi/pyversions/stakeout-agent" alt="Python versions">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
  </a>
  <a href="https://github.com/KyriakosFrang/stakeout-agent/actions/workflows/python-package.yml">
    <img src="https://github.com/KyriakosFrang/stakeout-agent/actions/workflows/python-package.yml/badge.svg" alt="CI">
  </a>
  <a href="https://github.com/astral-sh/uv">
    <img src="https://img.shields.io/badge/package%20manager-uv-8A2BE2" alt="uv">
  </a>
  <a href="https://github.com/astral-sh/ruff">
    <img src="https://img.shields.io/badge/linting-ruff-261230" alt="Ruff">
  </a>
</p>


![Dashboard timeline view](https://github.com/KyriakosFrang/stakeout-agent/blob/main/stakeout-agent/public/image.png?raw=true)

---

## Install and go

```bash
pip install stakeout-agent
```

```python
from stakeout_agent import LangGraphMonitorCallback

monitor = LangGraphMonitorCallback(graph_id="my_graph", thread_id="thread_123")
result = graph.invoke(inputs, config={"callbacks": [monitor]})
```

That's it. Every node execution, tool call, latency, and error is now in your database.

---

## How it works
```mermaid
graph LR
    A[Your LangGraph / CrewAI app] -->|callback| B[stakeout-agent]
    B --> C[(MongoDB)]
    B --> D[(PostgreSQL)]
    C --> E[Dashboard / your queries]
    D --> E
```

stakeout-agent hooks into your framework's event system. It records a `run` document for each invocation and an `event` document for every node start/end, tool call, tool result, and error — with latency tracked at every step.

---

## Why stakeout-agent?

| | stakeout-agent |
|---|---|
| Lines of integration code | **3** |
| Crashes your app on DB failure | **Never** — errors are logged, not raised |
| Node-level latency (P95) | **Yes** — tracked per node and per tool |
| Frameworks | **LangGraph + CrewAI** |
| Backends | **MongoDB + PostgreSQL** |
| Dashboard included | **Yes** — Streamlit, zero config |

---

## Installation

```bash
# MongoDB backend (default)
pip install stakeout-agent

# PostgreSQL backend
pip install 'stakeout-agent[postgres]'

# CrewAI support
pip install 'stakeout-agent[crewai]'
```

Requires Python 3.10+.

---

## Quick start

### LangGraph — Sync

```python
from stakeout_agent import LangGraphMonitorCallback

monitor = LangGraphMonitorCallback(graph_id="my_graph", thread_id="thread_123")
result = graph.invoke(inputs, config={"callbacks": [monitor]})
```

### LangGraph — Async

```python
from stakeout_agent import AsyncLangGraphMonitorCallback

monitor = AsyncLangGraphMonitorCallback(graph_id="my_graph", thread_id="thread_123")
result = await graph.ainvoke(inputs, config={"callbacks": [monitor]})
```

### CrewAI — Sync

```python
from stakeout_agent import CrewAIMonitorCallback

monitor = CrewAIMonitorCallback(crew_id="my_crew", thread_id="thread_123")
crew.kickoff(inputs={...})
```

`CrewAIMonitorCallback` registers itself with CrewAI's event bus automatically — no extra wiring needed.

### CrewAI — Async

```python
from stakeout_agent import AsyncCrewAIMonitorCallback

monitor = AsyncCrewAIMonitorCallback(crew_id="my_crew", thread_id="thread_123")
await crew.akickoff(inputs={...})
```

---

## Dashboard

Visualise runs, node timelines, and tool call details with the included Streamlit dashboard:

```bash
docker compose up -d mongo
cd stakeout-agent
uv run python examples/seed_demo_data.py   # optional: load demo data
uv run --with streamlit streamlit run examples/dashboard.py
```

Open `http://localhost:8501`. The dashboard shows:

- **Run History** — recent runs, status, duration, and a runs-over-time chart
- **Node Performance** — average and P95 latency per node and tool, error counts
- **Run Inspector** — full event timeline for any individual run
- **Thread Deep Dive** — multi-turn conversation view across all runs in a thread

---

## Try the examples

### LangGraph

A self-contained example that requires no LLM API key — nodes are pure Python functions.

```bash
docker compose up -d mongo
cd stakeout-agent
uv run python examples/dummy_app.py
```

### CrewAI

Requires a running MongoDB instance and an OpenAI API key (or configure a different provider via the `llm` parameter on each `Agent`).

**Sync:**

```bash
docker compose up -d mongo
cd stakeout-agent
OPENAI_API_KEY=sk-... uv run --with crewai python examples/dummy_crewai_app.py
```

**Async:**

```bash
docker compose up -d mongo
cd stakeout-agent
OPENAI_API_KEY=sk-... uv run --with crewai python examples/dummy_crewai_async_app.py
```

Each example runs a two-agent crew (Researcher + Writer) with a `MultiplyTool`, then prints the `runs` and `events` documents written to MongoDB.

---

## Configuration

| Environment variable | Default | Description |
|---|---|---|
| `STAKEOUT_BACKEND` | `mongodb` | Backend to use: `mongodb` or `postgres` |
| `MONGO_URI` | `mongodb://localhost:27017` | MongoDB connection string |
| `MONGO_DB` | `stakeout` | MongoDB database name |
| `POSTGRES_URI` | `postgresql://localhost/stakeout` | PostgreSQL connection string (also reads `DATABASE_URL`) |

### PostgreSQL

```bash
export STAKEOUT_BACKEND=postgres
export POSTGRES_URI=postgresql://user:password@localhost/stakeout
```

Tables are created automatically on first connection — no migration needed.

```bash
docker compose up -d postgres
# connection string: postgresql://stakeout:stakeout@localhost/stakeout
```

You can also inject a backend instance directly:

```python
from stakeout_agent import LangGraphMonitorCallback, PostgresMonitorDB

monitor = LangGraphMonitorCallback(
    graph_id="my_graph",
    thread_id="thread_123",
    db=PostgresMonitorDB(),
)
```

---

## What gets recorded

### `runs`

One document per graph/crew invocation.

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

### `events`

One document per node/task start/end, tool call, or error.

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
| `node_start` | A graph node or crew task begins | absent |
| `node_end` | A graph node or crew task completes | present |
| `tool_call` | A tool is invoked | absent |
| `tool_result` | A tool returns a result | present |
| `error` | A node, task, or tool raises an exception | present |

---

## Error handling

All database writes catch exceptions and log them — a monitoring failure will never crash your application. Enable `DEBUG` logging to see them:

```python
import logging
logging.getLogger("stakeout_agent").setLevel(logging.DEBUG)
```

---

## Querying the database directly

### MongoDB

```python
from stakeout_agent import MongoMonitorDB

db = MongoMonitorDB()
runs = list(db.runs.find({"graph_id": "my_graph"}).sort("started_at", -1))
events = list(db.events.find({"run_id": "<run_id>"}).sort("timestamp", 1))
```

### PostgreSQL

```python
import psycopg2

conn = psycopg2.connect("postgresql://user:password@localhost/stakeout")
with conn.cursor() as cur:
    cur.execute("SELECT * FROM runs WHERE graph_id = %s ORDER BY started_at DESC", ("my_graph",))
    runs = cur.fetchall()
```

---

## Extending stakeout-agent

**New framework:** create a file under `callback_handler/` that inherits `_MonitorBase` and implements the target framework's callback protocol.

**New database:** create a class that inherits `AbstractMonitorDB` and implement `create_run`, `complete_run`, `fail_run`, and `insert_event`.

```
stakeout_agent/
├── backends/
│   ├── base.py        # AbstractMonitorDB — shared interface
│   ├── mongodb.py     # MongoMonitorDB
│   ├── postgres.py    # PostgresMonitorDB
│   └── __init__.py    # get_backend() factory
├── callback_handler/
│   ├── base.py        # _MonitorBase — framework-agnostic core logic
│   ├── langgraph.py   # LangGraphMonitorCallback, AsyncLangGraphMonitorCallback
│   ├── crewai.py      # CrewAIMonitorCallback, AsyncCrewAIMonitorCallback
│   └── __init__.py
```

---

## Roadmap

- [x] Sync LangGraph callback support
- [x] Async LangGraph callback support
- [x] Sync CrewAI callback support
- [x] Async CrewAI callback support
- [x] MongoDB persistence
- [x] PostgreSQL persistence
- [x] Run and event collections
- [x] Streamlit dashboard (Run History, Node Performance, Run Inspector, Thread Deep Dive)
- [ ] Additional agentic frameworks (PydanticAI, SemanticKernel, AutoGen etc.)
- [ ] Additional storage backends (SQLite, Redis, ...)

## License

MIT
