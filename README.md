# LangGraph Monitor

Self-hosted monitoring for LangGraph applications. Drop a single callback into any graph and get a live dashboard showing runs, node timelines, tool calls, errors, and latency — no LangSmith required.

## Quick start (demo)

```bash
docker compose up --build
```

- Dashboard: http://localhost:3000
- API docs: http://localhost:8000/docs

The `graph-runner` service runs a dummy LangGraph graph every 3–8 seconds to generate data.

---

## Add monitoring to your own graph

### 1. Install

```bash
pip install git+https://github.com/your-org/langgraph-monitor.git#subdirectory=stakeout-agent
```

Or from a local clone:

```bash
pip install ./stakeout-agent
```

### 2. Start the monitoring stack

```bash
docker compose up mongo api dashboard -d
```

### 3. Add the callback

```python
from stakeout_agent import LangGraphMonitorCallback

monitor = LangGraphMonitorCallback(
    graph_id="my-graph",
    thread_id="thread-123",
)

result = graph.invoke(
    inputs,
    config={"callbacks": [monitor]},
)
```

Open http://localhost:3000 to see the run.

---

## What gets captured

| Event | Data |
|---|---|
| Graph start / end | run ID, graph ID, thread ID, status, timestamps |
| Node start / end | node name, inputs, outputs, latency |
| Tool call / result | tool name, input string, output, latency |
| Errors | node or tool name, exception type and message |

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `MONGO_URI` | `mongodb://localhost:27017` | MongoDB connection string |
| `MONGO_DB` | `stakeout` | Database name |

Both are read from the environment at first use, so you can configure them before the first graph invocation.

---

## API endpoints

| Endpoint | Description |
|---|---|
| `GET /runs` | List runs (filter by `graph_id`, `status`) |
| `GET /runs/{run_id}` | Single run detail |
| `GET /runs/{run_id}/events` | All events for a run |
| `GET /stats` | Aggregate stats across all graphs |
| `GET /graphs` | Per-graph summary |

---

## Stack

| Component | Technology |
|---|---|
| `stakeout-agent` package | Python, LangChain callbacks |
| Storage | MongoDB |
| API | FastAPI |
| Dashboard | React + Vite |
