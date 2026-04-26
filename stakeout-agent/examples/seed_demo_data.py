"""
Seed MongoDB with realistic demo data for the Stakeout Agent dashboard.

Usage:
    docker compose up -d mongo
    uv run python examples/seed_demo_data.py
"""

import random
import uuid
from datetime import datetime, timedelta, timezone

from pymongo import MongoClient

MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "stakeout"

GRAPHS = [
    {
        "graph_id": "customer_support_agent",
        "nodes": ["classify_intent", "retrieve_context", "draft_reply", "quality_check"],
        "tools": ["search_knowledge_base", "lookup_order"],
    },
    {
        "graph_id": "code_review_agent",
        "nodes": ["parse_diff", "analyse_style", "check_security", "summarise"],
        "tools": ["run_linter", "search_docs"],
    },
    {
        "graph_id": "research_agent",
        "nodes": ["plan_queries", "web_search", "synthesise", "format_report"],
        "tools": ["web_search_tool", "arxiv_lookup"],
    },
]

NODE_LATENCIES = {
    "classify_intent": (40, 15),
    "retrieve_context": (120, 40),
    "draft_reply": (800, 200),
    "quality_check": (600, 150),
    "parse_diff": (50, 10),
    "analyse_style": (300, 80),
    "check_security": (450, 120),
    "summarise": (700, 180),
    "plan_queries": (200, 60),
    "web_search": (900, 300),
    "synthesise": (1100, 250),
    "format_report": (400, 100),
}

TOOL_LATENCIES = {
    "search_knowledge_base": (80, 30),
    "lookup_order": (40, 15),
    "run_linter": (200, 60),
    "search_docs": (70, 25),
    "web_search_tool": (650, 200),
    "arxiv_lookup": (400, 120),
}

TOOL_EXAMPLES = {
    "search_knowledge_base": {
        "inputs": [
            'query: "return policy damaged items"',
            'query: "refund processing time"',
            'query: "shipping delay compensation"',
        ],
        "outputs": [
            "Policy: Items can be returned within 30 days. Damaged items qualify for immediate replacement.",
            "Refunds are processed within 3-5 business days after the return is received.",
            "For delays over 7 days, customers are eligible for a 10% discount on their next order.",
        ],
    },
    "lookup_order": {
        "inputs": ["order_id: ORD-8821", "order_id: ORD-4492", "order_id: ORD-1173"],
        "outputs": [
            "ORD-8821: shipped 2025-04-20, carrier=UPS, tracking=1Z999AA1, status=in_transit",
            "ORD-4492: delivered 2025-04-22, carrier=FedEx, tracking=789123456",
            "ORD-1173: processing, estimated ship date=2025-04-27",
        ],
    },
    "run_linter": {
        "inputs": ["file: auth_middleware.py", "file: db_pool.py", "file: rate_limiter.py"],
        "outputs": [
            "2 warnings: line 47 unused import, line 89 line too long (124 chars)",
            "All checks passed. No issues found.",
            "1 error: line 23 undefined variable 'redis_client'",
        ],
    },
    "search_docs": {
        "inputs": [
            'query: "sqlalchemy connection pool best practices"',
            'query: "jwt token expiry handling"',
        ],
        "outputs": [
            "SQLAlchemy docs: pool_size=5, max_overflow=10, pool_timeout=30 are recommended defaults.",
            "JWT best practice: set exp claim, validate on every request, use short-lived tokens (15 min).",
        ],
    },
    "web_search_tool": {
        "inputs": [
            'query: "RAG production 2025 survey"',
            'query: "LLM hallucination benchmark 2025"',
            'query: "code generation LLM HumanEval 2025"',
        ],
        "outputs": [
            "Found 12 papers. Top result: 'RAG in Production: Lessons from 50 deployments' (2025-03). Key themes: hybrid retrieval, re-ranking, eval pipelines.",
            "Found 8 papers. Top result: 'HalluBench 2025: A Comprehensive Hallucination Benchmark'. GPT-4o leads at 91.2% factuality.",
            "Found 15 papers. HumanEval+ leaderboard: Claude 3.7=94.1%, GPT-4o=93.8%, Gemini 2.0=91.4%.",
        ],
    },
    "arxiv_lookup": {
        "inputs": [
            'title: "Self-RAG: Learning to Retrieve, Generate, and Critique"',
            'title: "Constitutional AI: Harmlessness from AI Feedback"',
        ],
        "outputs": [
            "arXiv:2310.11511 — Self-RAG introduces adaptive retrieval with critique tokens. 318 citations.",
            "arXiv:2212.08073 — Constitutional AI describes RLAIF. 1,204 citations.",
        ],
    },
}

# Per-graph conversation templates: (human_msg, ai_response)
CONVERSATIONS = {
    "customer_support_agent": [
        (
            "My order #ORD-8821 hasn't arrived yet — it's been two weeks. Can you help?",
            "I looked up order #ORD-8821 and can see it shipped on April 20th via UPS. "
            "It's currently in transit (tracking: 1Z999AA1). The expected delivery is April 29th. "
            "If it doesn't arrive by then, I can file a claim on your behalf. Is there anything else I can help with?",
        ),
        (
            "I received a damaged item in my last order. How do I return it?",
            "I'm sorry to hear that! Damaged items qualify for an immediate return under our 30-day policy. "
            "I've created a pre-paid return label — you'll receive it by email shortly. "
            "Once we receive the item, a replacement will be dispatched within 24 hours.",
        ),
        (
            "I was charged twice for order #ORD-4492. Can you investigate?",
            "I can confirm there were two charges for order #ORD-4492 — this appears to have been a payment processor error. "
            "I've issued a full refund of the duplicate charge. "
            "You should see the credit on your statement within 3-5 business days.",
        ),
        (
            "What's your refund policy for digital products?",
            "Digital products are eligible for a refund within 14 days of purchase if they haven't been downloaded or accessed. "
            "Once downloaded, refunds are assessed case-by-case. "
            "If you'd like to request one, I can raise it to our billing team for review.",
        ),
        (
            "My package has been stuck in transit for 5 days according to the tracking page.",
            "I checked the carrier's network and there appears to be a sorting facility delay affecting your region. "
            "Your package is still in the system and should clear within 48 hours. "
            "As compensation for the inconvenience, I've applied a 10% discount to your account for your next order.",
        ),
    ],
    "code_review_agent": [
        (
            "Please review this PR: Added input validation to the user registration endpoint.",
            "Review complete. Two issues found:\n"
            "1. **Line 47** — the email regex allows empty strings. Replace with `r'^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$'`.\n"
            "2. **Line 89** — `username` is not sanitised before being passed to the ORM query (potential SQLi). Use parameterised queries.\n"
            "Everything else looks good. LGTM after fixes.",
        ),
        (
            "Check this diff: Refactored database connection pooling logic.",
            "The connection pool refactor looks solid overall. One suggestion:\n"
            "Consider adding a `pool_pre_ping=True` flag to detect stale connections before checkout — "
            "this prevents `OperationalError` spikes after a database restart. "
            "No blocking issues. Approved.",
        ),
        (
            "Review my rate limiting middleware implementation.",
            "Rate limiting logic is correct. The Redis key expiry matches the window size. One concern:\n"
            "The 429 error response body includes the rate limit threshold (`X-RateLimit-Limit: 100`). "
            "This leaks implementation details — consider stripping that header or making it configurable. "
            "Minor: add a `Retry-After` header so clients can back off gracefully.",
        ),
        (
            "Added JWT authentication middleware — can you review?",
            "JWT middleware review:\n"
            "- Token expiry validation is correct (`exp` claim checked against UTC now).\n"
            "- **Issue**: The secret is read from `os.environ['SECRET']` at import time. "
            "If the env var is missing the app crashes silently at startup. Use a startup health check.\n"
            "- The algorithm is hardcoded to HS256 — consider making it configurable for RS256 in production.",
        ),
    ],
    "research_agent": [
        (
            "What are the latest production-ready advances in RAG systems?",
            "Based on my research (12 papers, April 2025):\n\n"
            "**Top trends in production RAG:**\n"
            "1. **Hybrid retrieval** — combining dense (embedding) + sparse (BM25) consistently outperforms either alone by 8–12% on BEIR.\n"
            "2. **Re-ranking** — a cross-encoder re-ranker after initial retrieval reduces hallucination by ~15%.\n"
            "3. **Self-RAG** — models that decide *when* to retrieve outperform always-retrieve baselines.\n\n"
            "Key paper: *'RAG in Production: Lessons from 50 Deployments'* (arXiv, March 2025).",
        ),
        (
            "Summarise recent work on LLM hallucination mitigation techniques.",
            "Summary of hallucination mitigation research (2024–2025):\n\n"
            "1. **Factuality-aware RLHF** — training signal penalises fabricated citations. GPT-4o reaches 91.2% on HalluBench.\n"
            "2. **RAG + citation grounding** — forcing the model to cite retrieved passages reduces hallucination by up to 40%.\n"
            "3. **Chain-of-verification (CoVe)** — model generates, then self-critiques claims via follow-up queries.\n\n"
            "The field is moving toward multi-stage pipelines over single inference-time fixes.",
        ),
        (
            "What is the current state of the art for code generation with LLMs?",
            "Code generation SotA as of April 2025 (HumanEval+):\n\n"
            "| Model | Pass@1 |\n"
            "|---|---|\n"
            "| Claude 3.7 Sonnet | 94.1% |\n"
            "| GPT-4o | 93.8% |\n"
            "| Gemini 2.0 Flash | 91.4% |\n\n"
            "Key techniques: speculative decoding for speed, execution-based self-repair (model runs tests and fixes failures), "
            "and retrieval-augmented generation over internal codebases.",
        ),
    ],
}

THREADS = [f"thread_{i:03d}" for i in range(1, 16)]


def _latency(name: str, store: dict) -> float:
    mu, sigma = store.get(name, (200, 80))
    return max(10.0, round(random.gauss(mu, sigma), 1))


def _tool_example(tool_name: str) -> tuple[str, str]:
    examples = TOOL_EXAMPLES.get(tool_name, {})
    inp = random.choice(examples.get("inputs", [f"query from {tool_name}"]))
    out = random.choice(examples.get("outputs", [f"<result from {tool_name}>"]))
    return inp, out


def seed(num_runs: int = 80, days_back: int = 7):
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    db.runs.drop()
    db.events.drop()

    now = datetime.now(timezone.utc)
    runs_bulk = []
    events_bulk = []

    # Track per-thread conversation history so multi-turn threads accumulate context.
    thread_histories: dict[str, list[dict]] = {}
    # Assign each thread a fixed graph so the conversation stays coherent.
    thread_graph: dict[str, dict] = {}

    for _ in range(num_runs):
        thread_id = random.choice(THREADS)

        if thread_id not in thread_graph:
            thread_graph[thread_id] = random.choice(GRAPHS)
        graph_spec = thread_graph[thread_id]

        graph_id = graph_spec["graph_id"]
        nodes = graph_spec["nodes"]
        tools = graph_spec["tools"]

        # Pick a conversation turn for this thread (cycle through available turns)
        history = thread_histories.get(thread_id, [])
        turns = CONVERSATIONS[graph_id]
        turn_index = len([m for m in history if m["role"] == "human"]) % len(turns)
        human_content, ai_content = turns[turn_index]

        history_with_human = history + [{"role": "human", "content": human_content}]
        history_with_ai = history_with_human + [{"role": "assistant", "content": ai_content}]

        run_id = str(uuid.uuid4())
        started_at = now - timedelta(
            days=random.uniform(0, days_back),
            hours=random.uniform(0, 23),
            minutes=random.uniform(0, 59),
        )

        fail_at_node = random.randint(1, len(nodes)) if random.random() < 0.12 else None

        cursor = started_at
        run_events = []
        is_first_node = True

        for i, node in enumerate(nodes):
            fail_here = fail_at_node is not None and i == fail_at_node - 1
            is_last_node = i == len(nodes) - 1

            run_events.append(
                {
                    "run_id": run_id,
                    "graph_id": graph_id,
                    "event_type": "node_start",
                    "node_name": node,
                    "timestamp": cursor,
                    "payload": {"inputs": f"<state at {node}>"},
                    "messages": history_with_human if is_first_node else None,
                    "latency_ms": None,
                    "error": None,
                }
            )
            is_first_node = False
            node_lat = _latency(node, NODE_LATENCIES)
            cursor += timedelta(milliseconds=node_lat)

            if random.random() < 0.5 and not fail_here:
                tool = random.choice(tools)
                tool_input, tool_output = _tool_example(tool)
                run_events.append(
                    {
                        "run_id": run_id,
                        "graph_id": graph_id,
                        "event_type": "tool_call",
                        "node_name": tool,
                        "timestamp": cursor,
                        "payload": {"input": tool_input},
                        "latency_ms": None,
                        "error": None,
                    }
                )
                tool_lat = _latency(tool, TOOL_LATENCIES)
                cursor += timedelta(milliseconds=tool_lat)
                run_events.append(
                    {
                        "run_id": run_id,
                        "graph_id": graph_id,
                        "event_type": "tool_result",
                        "node_name": tool,
                        "timestamp": cursor,
                        "payload": {"output": tool_output},
                        "latency_ms": round(tool_lat, 1),
                        "error": None,
                    }
                )

            if fail_here:
                run_events.append(
                    {
                        "run_id": run_id,
                        "graph_id": graph_id,
                        "event_type": "error",
                        "node_name": node,
                        "timestamp": cursor,
                        "payload": {},
                        "latency_ms": round(node_lat, 1),
                        "error": f"RuntimeError: unexpected output from {node}",
                    }
                )
                break

            run_events.append(
                {
                    "run_id": run_id,
                    "graph_id": graph_id,
                    "event_type": "node_end",
                    "node_name": node,
                    "timestamp": cursor,
                    "payload": {"outputs": f"<output from {node}>"},
                    "messages": history_with_ai if is_last_node else None,
                    "latency_ms": round(node_lat, 1),
                    "error": None,
                }
            )

        ended_at = cursor
        failed = fail_at_node is not None

        if not failed:
            thread_histories[thread_id] = history_with_ai

        run_doc = {
            "_id": run_id,
            "graph_id": graph_id,
            "thread_id": thread_id,
            "status": "failed" if failed else "completed",
            "started_at": started_at,
            "ended_at": ended_at,
            "error": run_events[-1]["error"] if failed else None,
            "metadata": {},
        }
        runs_bulk.append(run_doc)
        events_bulk.extend(run_events)

    db.runs.insert_many(runs_bulk)
    db.events.insert_many(events_bulk)

    print(f"Seeded {len(runs_bulk)} runs and {len(events_bulk)} events into '{DB_NAME}'.")
    failed_count = sum(1 for r in runs_bulk if r["status"] == "failed")
    print(f"  completed={len(runs_bulk) - failed_count}  failed={failed_count}")
    print(f"  threads with history: {len(thread_histories)}")


if __name__ == "__main__":
    seed()
