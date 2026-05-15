<h1 align="center">stakeout-agent</h1>

<p align="center">
  <strong>Drop-in observability for LangGraph and CrewAI.</strong>
</p>

<p align="center">
   One callback. Every run, node, tool call, token count, prompt, and response — captured automatically into MongoDB, PostgreSQL, or any OpenTelemetry-compatible collector. No changes to your agent code.
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
# LangGraph + MongoDB
pip install 'stakeout-agent[langgraph,mongodb]'

# LangGraph + PostgreSQL
pip install 'stakeout-agent[langgraph,postgres]'

# LangGraph + OpenTelemetry (Jaeger, Datadog, Grafana Tempo, Honeycomb, …)
pip install 'stakeout-agent[langgraph,otel]'

# CrewAI + MongoDB
pip install 'stakeout-agent[crewai,mongodb]'

# CrewAI + PostgreSQL
pip install 'stakeout-agent[crewai,postgres]'

# CrewAI + OpenTelemetry
pip install 'stakeout-agent[crewai,otel]'
```

```python
from stakeout_agent import LangGraphMonitorCallback

monitor = LangGraphMonitorCallback(graph_id="my_graph", thread_id="thread_123")
result = graph.invoke(inputs, config={"callbacks": [monitor]})
```

That's it. Every node execution, tool call, latency, token count, prompt, response, and error is now in your database.

---

## How it works
```mermaid
graph LR
    A[Your LangGraph / CrewAI app] -->|callback| B[stakeout-agent]
    B --> C[(MongoDB)]
    B --> D[(PostgreSQL)]
    B --> F[OTEL Collector]
    C --> E[Dashboard / your queries]
    D --> E
    F --> G[Jaeger / Datadog / Grafana / Honeycomb]
```

stakeout-agent hooks into your framework's event system. It records a `run` document for each invocation and an `event` document for every node start/end, tool call, tool result, and error — with latency, token usage, and the actual prompts and responses captured at every step.

---

## Why stakeout-agent?

| | stakeout-agent |
|---|---|
| Lines of integration code | **3** |
| Crashes your app on DB failure | **Never** — errors are logged, not raised |
| Node-level latency (P95) | **Yes** — tracked per node and per tool |
| Token usage | **Yes** — per node and rolled up to the run |
| Cost estimation | **Yes** — opt-in, configurable per model |
| Prompt & response capture | **Yes** — per node, opt-out, truncation supported |
| Frameworks | **LangGraph + CrewAI** |
| Backends | **MongoDB + PostgreSQL + OpenTelemetry** |
| Dashboard included | **Yes** — [dedicated real-time observability UI](https://github.com/KyriakosFrang/stakeout-dashboard) |

---

## Installation

Install only what you need — framework and backend are independent extras:

```bash
# LangGraph + MongoDB
pip install 'stakeout-agent[langgraph,mongodb]'

# LangGraph + PostgreSQL
pip install 'stakeout-agent[langgraph,postgres]'

# LangGraph + OpenTelemetry (Jaeger, Datadog, Grafana Tempo, Honeycomb, …)
pip install 'stakeout-agent[langgraph,otel]'

# CrewAI + MongoDB
pip install 'stakeout-agent[crewai,mongodb]'

# CrewAI + PostgreSQL
pip install 'stakeout-agent[crewai,postgres]'

# CrewAI + OpenTelemetry
pip install 'stakeout-agent[crewai,otel]'
```

| Extra | Installs | Use when |
|---|---|---|
| `langgraph` | `langchain-core`, `langgraph` | Using LangGraph |
| `crewai` | `crewai` | Using CrewAI |
| `mongodb` | `pymongo` | Storing to MongoDB |
| `postgres` | `psycopg2-binary` | Storing to PostgreSQL |
| `otel` | `opentelemetry-sdk`, `opentelemetry-exporter-otlp-proto-grpc` | Exporting to any OTEL-compatible collector |

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

### Concurrent invocations

**LangGraph** — a single callback instance is safe to share across concurrent `graph.invoke()` / `graph.ainvoke()` calls. Per-run state is keyed internally by the root run UUID, so two simultaneous invocations never interfere with each other:

```python
# Fine — one shared instance, both runs tracked independently
monitor = AsyncLangGraphMonitorCallback(graph_id="g", thread_id="t")
await asyncio.gather(
    graph.ainvoke(inputs_a, config={"callbacks": [monitor]}),
    graph.ainvoke(inputs_b, config={"callbacks": [monitor]}),
)
```

**CrewAI** — use a separate instance per concurrent kickoff. CrewAI's event bus is global and does not carry a run identifier, so events from two simultaneous crews on the same listener cannot be reliably routed:

```python
# Correct — separate instance per concurrent kickoff
await asyncio.gather(
    crew.akickoff(inputs_a),   # monitor_a registered with crew_a
    crew.akickoff(inputs_b),   # monitor_b registered with crew_b
)
```

---

## Token usage and cost tracking

Token counts are captured automatically from every LLM call — no changes to your agent code required. Per-node input/output tokens are recorded on each `node_end` event, and totals are rolled up onto the `run` document at completion.

### Token capture only (always on)

```python
from stakeout_agent import LangGraphMonitorCallback

monitor = LangGraphMonitorCallback(graph_id="my_graph", thread_id="thread_123")
result = graph.invoke(inputs, config={"callbacks": [monitor]})
```

Token fields (`input_tokens`, `output_tokens`, `model`) appear on `node_end` events and `total_input_tokens` / `total_output_tokens` on the run document whenever the LLM response contains usage metadata.

Cache token fields (`cache_read_tokens`, `cache_creation_tokens`) are captured automatically for providers that report them — Anthropic (prompt caching) and OpenAI (cached inputs). They appear on `node_end` events and roll up as `total_cache_read_tokens` / `total_cache_creation_tokens` on the run document.

### Cost estimation (opt-in)

```python
from stakeout_agent import LangGraphMonitorCallback
from stakeout_agent.pricing import ModelPricing, PricingMap

monitor = LangGraphMonitorCallback(
    graph_id="my_graph",
    thread_id="thread_123",
    pricing=PricingMap({
        "gpt-4o":      ModelPricing(input_cost_per_1k=0.005,   output_cost_per_1k=0.015),
        "gpt-4o-mini": ModelPricing(input_cost_per_1k=0.00015, output_cost_per_1k=0.0006),
    })
)
result = graph.invoke(inputs, config={"callbacks": [monitor]})
```

When `pricing` is provided, `estimated_cost_usd` is computed per LLM call and rolled up onto the run. Multi-model workflows are fully supported — each node resolves cost against the model it actually used. Models not present in the map are silently skipped; token counts are still recorded.

### Custom token extractor

The default extractor covers OpenAI (`token_usage` / `model_name`) and Anthropic (`usage` / `model`) response shapes. For providers with a different metadata structure, pass a `token_extractor`:

```python
def my_extractor(metadata: dict) -> tuple[int | None, int | None, str | None]:
    usage = metadata.get("llm_output", {}).get("token_usage", {})
    return usage.get("input"), usage.get("output"), metadata.get("model_id")

monitor = LangGraphMonitorCallback(
    graph_id="my_graph",
    thread_id="thread_123",
    token_extractor=my_extractor,
)
```

The extractor receives `response.llm_output` and must return `(input_tokens, output_tokens, model_name)`. Any field can be `None`.

---

## Prompt and response capture

The exact messages sent to the LLM and the response text are captured automatically on each `node_end` event. This is on by default and requires no configuration.

```python
from stakeout_agent import LangGraphMonitorCallback

monitor = LangGraphMonitorCallback(graph_id="my_graph", thread_id="thread_123")
result = graph.invoke(inputs, config={"callbacks": [monitor]})
```

Each `node_end` event will include:

```json
{
  "event_type": "node_end",
  "node_name": "agent",
  "llm_input": [
    { "role": "system", "content": "You are a helpful assistant." },
    { "role": "user",   "content": "Summarize the following document..." }
  ],
  "llm_output": "Here is a concise summary..."
}
```

`llm_input` and `llm_output` are absent when no LLM call occurred within the node (e.g. pure routing nodes).

### Opt out for sensitive workloads

```python
monitor = LangGraphMonitorCallback(
    graph_id="my_graph",
    thread_id="thread_123",
    capture_payloads=False,
)
```

Recommended for regulated or privacy-sensitive environments (financial services, healthcare) where prompt content may include PII or confidential data.

### Limit stored content size

```python
monitor = LangGraphMonitorCallback(
    graph_id="my_graph",
    thread_id="thread_123",
    max_payload_chars=2000,
)
```

Each message's content and the response text are truncated to `max_payload_chars` characters before storage. Useful for long-context or multi-turn workflows to prevent unbounded document sizes.

Both options apply identically to `AsyncLangGraphMonitorCallback`, `CrewAIMonitorCallback`, and `AsyncCrewAIMonitorCallback`.

---

## Dashboard

A dedicated dashboard repository is available at **[stakeout-dashboard](https://github.com/KyriakosFrang/stakeout-dashboard)** — a standalone Streamlit app that connects to your MongoDB or PostgreSQL backend and visualises everything stakeout-agent captures.

The dashboard shows:

- **Run History** — recent runs, status, duration, and a runs-over-time chart
- **Node Performance** — average and P95 latency per node and tool, error counts
- **Run Inspector** — full event timeline for any individual run
- **Thread Deep Dive** — multi-turn conversation view across all runs in a thread

See the [stakeout-dashboard README](https://github.com/KyriakosFrang/stakeout-dashboard) for setup and configuration instructions.

---

## Try the examples

### LangGraph

A self-contained example that requires no LLM API key — nodes are pure Python functions.

```bash
docker compose up -d mongo
cd stakeout-agent
uv run --extra langgraph --extra mongodb python examples/dummy_app.py
```

### CrewAI

Requires a running MongoDB instance and an OpenAI API key (or configure a different provider via the `llm` parameter on each `Agent`).

**Sync:**

```bash
docker compose up -d mongo
cd stakeout-agent
OPENAI_API_KEY=sk-... uv run --extra crewai --extra mongodb python examples/dummy_crewai_app.py
```

**Async:**

```bash
docker compose up -d mongo
cd stakeout-agent
OPENAI_API_KEY=sk-... uv run --extra crewai --extra mongodb python examples/dummy_crewai_async_app.py
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
| `OTEL_EXPORTER_OTLP_ENDPOINT` | — | OTLP collector endpoint; triggers auto-configure when set |
| `OTEL_EXPORTER_OTLP_HEADERS` | — | Headers for the OTLP exporter (e.g. auth tokens) |
| `OTEL_SERVICE_NAME` | `stakeout-agent` | Service name attached to all spans |

### PostgreSQL

```bash
export STAKEOUT_BACKEND=postgres
export POSTGRES_URI=postgresql://user:password@localhost/stakeout
```

Tables are created automatically on first connection — no migration needed. New columns (`llm_input`, `llm_output`, token and cost fields) are added to existing tables via `ALTER TABLE … ADD COLUMN IF NOT EXISTS`.

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

### OpenTelemetry

Export every run and node as an OTEL trace to any compatible collector — Jaeger, Datadog, Grafana Tempo, Honeycomb, and others — without changing your agent code.

```bash
pip install 'stakeout-agent[langgraph,otel]'
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
export OTEL_SERVICE_NAME=my-agent-service   # optional, defaults to "stakeout-agent"
```

```python
from stakeout_agent import LangGraphMonitorCallback
from stakeout_agent.backends.otel import OTELMonitorDB

monitor = LangGraphMonitorCallback(
    graph_id="my_graph",
    thread_id="thread_123",
    db=OTELMonitorDB(),  # reads OTEL_EXPORTER_OTLP_ENDPOINT automatically
)
result = graph.invoke(inputs, config={"callbacks": [monitor]})
```

`OTELMonitorDB` honours the standard OTEL environment variables (`OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_EXPORTER_OTLP_HEADERS`, `OTEL_SERVICE_NAME`) so it drops into any existing OTEL setup with zero custom config.

For teams with a programmatic OTEL setup, inject your own `TracerProvider`:

```python
from stakeout_agent.backends.otel import OTELMonitorDB

monitor = LangGraphMonitorCallback(
    graph_id="my_graph",
    thread_id="thread_123",
    db=OTELMonitorDB(tracer_provider=my_provider),
)
```

If neither `OTEL_EXPORTER_OTLP_ENDPOINT` nor an explicit provider is given, `OTELMonitorDB` falls back to the global OTEL tracer provider configured elsewhere in your application.

#### Span structure

Each invocation produces a trace following [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/):

| stakeout concept | OTEL span |
|---|---|
| `run` | Root span — name is `graph_id` |
| `node_start` / `node_end` | Child span per node — name is the node name |
| `tool_call` / `tool_result` | Child span per tool call — name is the tool name |
| `retriever_start` / `retriever_end` | Child span per retriever call |
| `error` | `StatusCode.ERROR` + recorded exception on the relevant span |
| `latency_ms` | `stakeout.latency_ms` attribute (span duration also captures wall time) |
| `model` | `gen_ai.request.model` |
| `input_tokens` / `output_tokens` | `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens` |
| `cache_read_tokens` / `cache_creation_tokens` | `gen_ai.usage.cache_read_input_tokens` / `gen_ai.usage.cache_creation_input_tokens` |
| `estimated_cost_usd` | `stakeout.cost_usd` |
| `llm_input` / `llm_output` | Span events `gen_ai.content.prompt` / `gen_ai.content.completion` (not attributes, to avoid collector size limits) |
| `thread_id`, `graph_id`, `run_id` | `stakeout.thread_id`, `stakeout.graph_id`, `stakeout.run_id` on root span |

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
  "total_input_tokens": 1850,
  "total_output_tokens": 420,
  "estimated_cost_usd": 0.01553,
  "total_cache_read_tokens": 1200,
  "total_cache_creation_tokens": 650
}
```

`status` is one of `running`, `completed`, or `failed`. Token and cost fields are omitted when no LLM usage data is available; `estimated_cost_usd` is omitted when no `pricing` map is configured.

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
  "input_tokens": 320,
  "output_tokens": 85,
  "model": "gpt-4o",
  "llm_input": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Summarize the following document..."}
  ],
  "llm_output": "Here is a concise summary...",
  "payload": {"outputs": "..."},
  "error": null
}
```

| `event_type` | When | `latency_ms` | token fields | `llm_input` / `llm_output` |
|---|---|---|---|---|
| `node_start` | A graph node or crew task begins | absent | absent | absent |
| `node_end` | A graph node or crew task completes | present | present when LLM was called | present when LLM was called and `capture_payloads=True` |
| `tool_call` | A tool is invoked | absent | absent | absent |
| `tool_result` | A tool returns a result | present | absent | absent |
| `retriever_start` | A LangChain retriever starts (RAG) | absent | absent | absent |
| `retriever_end` | A retriever returns documents | present | absent | absent |
| `error` | A node, task, tool, or retriever raises an exception | present | absent | absent |

---

## Error handling

All database writes catch exceptions and log them — a monitoring failure will never crash your application. Enable `DEBUG` logging to see them:

```python
import logging
logging.getLogger("stakeout_agent").setLevel(logging.DEBUG)
```

---

## Threads and conversation history

### What `thread_id` means

`thread_id` is a label you assign to group related invocations together — typically a user session or a multi-turn conversation. stakeout-agent stores it on every run but does not manage it:

```
thread_id          ← your conversation identifier (you supply this)
  └── run_id       ← one graph.invoke() / crew.kickoff() call (generated per execution)
        └── events ← node_start, node_end, tool_call, tool_result, error
```

Every time you call `graph.invoke(...)` with the same `thread_id`, a new `run` is created under that thread. The `events` for each run are stored in order of `timestamp`.

### Viewing all steps in a conversation

To reconstruct the full execution history of a conversation, query runs by `thread_id` and then fetch events for each run in timestamp order.

**MongoDB:**

```python
from stakeout_agent import MongoMonitorDB

db = MongoMonitorDB()

thread_id = "thread_123"

runs = list(db.runs.find({"thread_id": thread_id}).sort("started_at", 1))
for run in runs:
    print(f"\n--- Run {run['_id']} ({run['status']}) ---")
    events = list(db.events.find({"run_id": run["_id"]}).sort("timestamp", 1))
    for ev in events:
        print(f"  [{ev['timestamp']}] {ev['event_type']:12s}  node={ev['node_name']}")
```

**PostgreSQL:**

```sql
SELECT r.run_id, e.timestamp, e.event_type, e.node_name, e.latency_ms, e.error
FROM events e
JOIN runs r ON r.run_id = e.run_id
WHERE r.thread_id = 'thread_123'
ORDER BY e.timestamp ASC;
```

The [stakeout-dashboard](https://github.com/KyriakosFrang/stakeout-dashboard) **Thread Deep Dive** view does exactly this — select any `thread_id` and see every run and every step in chronological order.


## Integration tests

Integration tests run against real backend services and are kept separate from the unit test suite so CI stays fast and dependency-free. Unit tests (mocks only) always run. Integration tests require Docker and are run locally or in a dedicated CI job.

### Test layout

| Path | Needs Docker | What it covers |
|---|---|---|
| `tests/` | No | All unit tests — mocked at the driver boundary |
| `tests/integration/test_mongo_integration.py` | `mongo` | Full CRUD lifecycle against a real MongoDB instance |
| `tests/integration/test_postgres_integration.py` | `postgres` | Full CRUD lifecycle against a real PostgreSQL instance |
| `tests/integration/test_otel_inprocess.py` | No | OTEL backend with `InMemorySpanExporter` — span tree, attributes, events, error paths |

The OTEL in-process tests use the SDK's `InMemorySpanExporter` and run without any container. MongoDB and Postgres tests auto-skip when the container isn't reachable, so a plain `pytest` never fails due to a missing service.

### Start the backends

```bash
# from the repo root
docker compose up -d
```

Wait for the healthchecks to pass (about 10–15 seconds), then confirm all three are healthy:

```bash
docker compose ps
```

| Service | Port | Notes |
|---|---|---|
| `stakeout-mongo` | `27017` | MongoDB 7 |
| `stakeout-postgres` | `5432` | PostgreSQL 16 — user/pass/db: `stakeout` |
| `stakeout-jaeger` | `4317` (OTLP gRPC), `16686` (UI) | Jaeger all-in-one |

### Run integration tests

```bash
cd stakeout-agent

# All backends at once
uv run --with pytest --extra langgraph --extra mongodb --extra postgres --extra crewai --extra otel pytest tests/integration -v

# One backend at a time
uv run --with pytest --extra mongodb pytest tests/integration/test_mongo_integration.py -v
uv run --with pytest --extra postgres pytest tests/integration/test_postgres_integration.py -v
uv run --with pytest --extra otel    pytest tests/integration/test_otel_inprocess.py -v
```

### Run only unit tests (no Docker)

```bash
cd stakeout-agent
uv run --with pytest --extra langgraph --extra mongodb --extra postgres --extra crewai --extra otel pytest --ignore tests/integration
```

### View OTEL traces in Jaeger

After running any code that uses `OTELMonitorDB` with `OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317`, open the Jaeger UI to browse traces:

```
http://localhost:16686
```

Select the `stakeout-agent` service (or whatever `OTEL_SERVICE_NAME` is set to) and explore the run timeline, node spans, tool calls, and token attributes.

---

## Roadmap

- [x] Sync LangGraph callback support
- [x] Async LangGraph callback support
- [x] Sync CrewAI callback support
- [x] Async CrewAI callback support
- [x] MongoDB persistence
- [x] PostgreSQL persistence
- [x] OpenTelemetry export (`OTELMonitorDB` — Jaeger, Datadog, Grafana Tempo, Honeycomb, …)
- [x] Run and event collections
- [x] Token usage tracking (per node and per run)
- [x] Cost estimation with configurable pricing map
- [x] Prompt and response capture per node (`capture_payloads`, `max_payload_chars`)
- [x] [Dedicated UI dashboard](https://github.com/KyriakosFrang/stakeout-dashboard) (Run History, Node Performance, Run Inspector, Thread Deep Dive)
- [ ] Additional agentic frameworks (PydanticAI, SemanticKernel, AutoGen etc.)
- [ ] Additional storage backends (SQLite, Redis, …)

## License

MIT
