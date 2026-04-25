# stakeout-agent

Drop-in monitoring for LangGraph applications. Captures every graph run, node execution, and tool call into MongoDB with no changes to your graph code.

## Why stakeout-agent?

When building LangGraph applications, understanding how your graphs execute is critical for debugging and optimization. stakeout-agent provides:

- **Zero code changes** — just add a callback to your graph config
- **Complete visibility** — captures node starts/ends, tool calls, and errors
- **MongoDB storage** — leverage your existing infrastructure
- **Framework-agnostic core** — easily extensible to other frameworks


## Installation

```bash
pip install stakeout-agent
```

Requires Python 3.10+ and a running MongoDB instance.

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

## Configuration

| Environment variable | Default | Description |
|---|---|---|
| `MONGO_URI` | `mongodb://localhost:27017` | MongoDB connection string |
| `MONGO_DB` | `stakeout` | Database name |

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

```json
{
  "run_id": "<run_id>",
  "graph_id": "my_graph",
  "event_type": "node_end",
  "node_name": "agent",
  "timestamp": "2026-04-25T10:00:03Z",
  "latency_ms": 1240.5,
  "payload": {},
  "error": null
}
```

| `event_type` | When |
|---|---|
| `node_start` | A graph node begins execution |
| `node_end` | A graph node completes |
| `tool_call` | A tool is invoked |
| `tool_result` | A tool returns a result |
| `error` | A node or tool raises an exception |

## Using `MonitorDB` directly

```python
from stakeout_agent import MonitorDB

db = MonitorDB()

# fetch all runs for a graph
runs = list(db.runs.find({"graph_id": "my_graph"}).sort("started_at", -1))

# fetch events for a specific run
events = list(db.events.find({"run_id": "<run_id>"}).sort("timestamp", 1))
```

## Package structure

```
stakeout_agent/
├── callback_handler/
│   ├── base.py        # _MonitorBase — framework-agnostic core logic
│   ├── langgraph.py   # LangGraphMonitorCallback, AsyncLangGraphMonitorCallback
│   └── __init__.py
└── db.py              # MonitorDB
```

Adding support for another framework means adding a single file under `callback_handler/` that inherits from `_MonitorBase` and implements the target framework's callback protocol.

## Possible Dashboard implementation
Utilizing the stored data, you could build a dashboard to visualize graph runs, node execution timelines, and tool call details. For example, a timeline view of node executions within a run could look like this:

![Alt text](https://github.com/KyriakosFrang/stakeout-agent/blob/main/stakeout-agent/public/image.png)

## License

MIT
