"""
Realistic LangGraph seed for MongoDB.

Runs three actual LangGraph graphs with a mock LLM (no API key required).
Every write goes through LangGraphMonitorCallback, so the documents in
MongoDB are byte-for-byte identical to what a production run would produce.

Usage:
    docker compose up -d mongo
    uv run python examples/langgraph_seed.py
    uv run python examples/langgraph_seed.py --runs 120
"""

from __future__ import annotations

import argparse
import random
from typing import Annotated, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pymongo import MongoClient

from stakeout_agent import LangGraphMonitorCallback
from stakeout_agent.pricing import ModelPricing, PricingMap

MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "stakeout"
THREADS = [f"thread_{i:03d}" for i in range(1, 16)]

PRICING = PricingMap({
    "claude-3-5-sonnet-20241022": ModelPricing(input_cost_per_1k=0.003, output_cost_per_1k=0.015),
    "claude-3-5-haiku-20241022": ModelPricing(input_cost_per_1k=0.0008, output_cost_per_1k=0.004),
    "gpt-4o": ModelPricing(input_cost_per_1k=0.0025, output_cost_per_1k=0.010),
})


# ---------------------------------------------------------------------------
# Mock LLM — cycles through canned responses, emits realistic token metadata
# ---------------------------------------------------------------------------

class _MockChatModel(BaseChatModel):
    responses: list[str]
    model_id: str = "claude-3-5-sonnet-20241022"
    mean_input_tokens: int = 600
    mean_output_tokens: int = 250

    @property
    def _llm_type(self) -> str:
        return "mock"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        response = random.choice(self.responses)
        in_tok = max(50, int(random.gauss(self.mean_input_tokens, self.mean_input_tokens * 0.15)))
        out_tok = max(20, int(random.gauss(self.mean_output_tokens, self.mean_output_tokens * 0.2)))
        ai_msg = AIMessage(content=response)
        return ChatResult(
            generations=[ChatGeneration(message=ai_msg, text=response)],
            llm_output={
                "model_name": self.model_id,
                "token_usage": {
                    "prompt_tokens": in_tok,
                    "completion_tokens": out_tok,
                    "total_tokens": in_tok + out_tok,
                },
            },
        )


# ---------------------------------------------------------------------------
# Tools  (invoke with config= so tool_call / tool_result callbacks fire)
# ---------------------------------------------------------------------------

@tool
def search_knowledge_base(query: str) -> str:
    """Search the customer support knowledge base."""
    return random.choice([
        "Items can be returned within 30 days. Damaged items qualify for immediate replacement.",
        "Refunds are processed within 3-5 business days after the return is received.",
        "For shipping delays over 7 days, customers are eligible for a 10% discount on next order.",
        "Digital products are refundable within 14 days if not downloaded or accessed.",
    ])


@tool
def lookup_order(order_id: str) -> str:
    """Look up order status by order ID."""
    records = {
        "ORD-8821": "shipped 2025-04-20, carrier=UPS, tracking=1Z999AA1, status=in_transit",
        "ORD-4492": "delivered 2025-04-22, carrier=FedEx, tracking=789123456, status=delivered",
        "ORD-1173": "processing, estimated ship date=2025-04-27",
        "ORD-2256": "shipped 2025-04-18, carrier=DHL, tracking=JD012345, status=delayed",
    }
    return records.get(order_id, f"Order {order_id} not found in system.")


@tool
def run_linter(file: str) -> str:
    """Run the project linter on a source file."""
    return random.choice([
        "2 warnings: line 47 unused import, line 89 line too long (124 chars)",
        "All checks passed. No issues found.",
        "1 error: line 23 undefined variable 'redis_client'",
        "3 warnings: missing docstrings on public methods (lines 12, 34, 67)",
    ])


@tool
def search_docs(query: str) -> str:
    """Search internal documentation."""
    return random.choice([
        "SQLAlchemy: pool_size=5, max_overflow=10, pool_timeout=30 are recommended defaults.",
        "JWT best practice: set exp claim, validate on every request, use short-lived tokens (15 min).",
        "Redis rate-limit pattern: INCR + EXPIRE in a pipeline; use Lua script for atomicity.",
    ])


@tool
def web_search_tool(query: str) -> str:
    """Search the web for recent papers and articles."""
    return random.choice([
        "Found 12 papers on RAG in production. Top: 'Lessons from 50 deployments' (2025-03). Key: hybrid retrieval, re-ranking.",
        "Found 8 papers on LLM hallucination. Top: HalluBench 2025. GPT-4o leads at 91.2% factuality.",
        "Found 15 papers on code generation. HumanEval+: Claude 3.7=94.1%, GPT-4o=93.8%.",
        "Found 10 papers on RLHF alignment. Reward model diversity cited as key driver.",
    ])


@tool
def arxiv_lookup(title: str) -> str:
    """Look up an arXiv paper by title."""
    return random.choice([
        "arXiv:2310.11511 — Self-RAG: adaptive retrieval with critique tokens. 318 citations.",
        "arXiv:2212.08073 — Constitutional AI: RLAIF. 1,204 citations.",
        "arXiv:2402.01030 — RAPTOR: recursive summarisation for long-document retrieval. 127 citations.",
    ])


# ---------------------------------------------------------------------------
# Graph 1: customer_support_agent
# ---------------------------------------------------------------------------

class SupportState(TypedDict):
    messages: Annotated[list, add_messages]
    context: str
    intent: str
    inject_error: str  # node name to fail at, or ""


_CS_CLASSIFIER = _MockChatModel(
    responses=[
        '{"intent": "order_status", "confidence": 0.96}',
        '{"intent": "return_request", "confidence": 0.91}',
        '{"intent": "refund_inquiry", "confidence": 0.88}',
        '{"intent": "general_question", "confidence": 0.79}',
    ],
    model_id="claude-3-5-sonnet-20241022",
    mean_input_tokens=320,
    mean_output_tokens=90,
)

_CS_DRAFTER = _MockChatModel(
    responses=[
        "I've looked up your order — it's currently in transit via UPS (tracking: 1Z999AA1), expected by April 29th. If it doesn't arrive, I can file a claim immediately.",
        "I'm sorry to hear your item arrived damaged. You qualify for an immediate replacement — I've created a pre-paid return label, please expect it by email within the hour.",
        "I can confirm two charges hit for your order — a payment processor error. I've issued a full refund of the duplicate charge; you'll see it within 3-5 business days.",
        "Digital products are refundable within 14 days if not downloaded. Since the file has been accessed, I'll escalate this to our billing team for a case-by-case review.",
        "There's a sorting-facility delay in your region. Your package is still in the system and should clear within 48 hours. I've applied a 10% discount to your account as compensation.",
    ],
    model_id="claude-3-5-sonnet-20241022",
    mean_input_tokens=950,
    mean_output_tokens=380,
)

_CS_QA = _MockChatModel(
    responses=[
        "The draft reply is accurate, empathetic, and actionable. No revisions needed.",
        "Revised: tone was slightly abrupt — updated to acknowledge the customer's frustration before stating the resolution.",
        "Reply approved. Correctly references the return policy and sets clear delivery expectations.",
    ],
    model_id="claude-3-5-sonnet-20241022",
    mean_input_tokens=1300,
    mean_output_tokens=180,
)

_CS_INPUTS = [
    "My order #ORD-8821 hasn't arrived yet — it's been two weeks. Can you help?",
    "I received a damaged item in my last order. How do I return it?",
    "I was charged twice for order #ORD-4492. Can you investigate?",
    "What's your refund policy for digital products?",
    "My package #ORD-2256 has been stuck in transit for 5 days.",
]


def _cs_classify_intent(state: SupportState, config: RunnableConfig) -> dict:
    if state.get("inject_error") == "classify_intent":
        raise RuntimeError("Simulated failure: classify_intent received malformed input")
    human_msg = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    reply = _CS_CLASSIFIER.invoke(
        [
            SystemMessage(content="Classify the customer intent. Return JSON with 'intent' and 'confidence'."),
            HumanMessage(content=human_msg),
        ],
        config=config,
    )
    return {"intent": reply.content, "messages": [reply]}


def _cs_retrieve_context(state: SupportState, config: RunnableConfig) -> dict:
    if state.get("inject_error") == "retrieve_context":
        raise RuntimeError("Simulated failure: knowledge base connection timed out")
    human_msg = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    order_id = next((w.rstrip(".?,") for w in human_msg.split() if w.startswith("ORD-")), None)
    if order_id:
        context = lookup_order.invoke({"order_id": order_id}, config=config)
    else:
        context = search_knowledge_base.invoke({"query": human_msg[:100]}, config=config)
    return {"context": context}


def _cs_draft_reply(state: SupportState, config: RunnableConfig) -> dict:
    if state.get("inject_error") == "draft_reply":
        raise RuntimeError("Simulated failure: LLM rate limit exceeded")
    human_msg = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    reply = _CS_DRAFTER.invoke(
        [
            SystemMessage(content="You are a helpful customer support agent. Draft a professional, empathetic response."),
            HumanMessage(content=f"Customer: {human_msg}\n\nContext: {state.get('context', '')}"),
        ],
        config=config,
    )
    return {"messages": [reply]}


def _cs_quality_check(state: SupportState, config: RunnableConfig) -> dict:
    if state.get("inject_error") == "quality_check":
        raise RuntimeError("Simulated failure: quality-check service timeout")
    draft = next((m.content for m in reversed(state["messages"]) if isinstance(m, AIMessage)), "")
    reply = _CS_QA.invoke(
        [
            SystemMessage(content="Review the draft reply for tone, accuracy, and completeness."),
            HumanMessage(content=f"Draft: {draft}"),
        ],
        config=config,
    )
    return {"messages": [reply]}


def build_customer_support_graph():
    g = StateGraph(SupportState)
    g.add_node("classify_intent", _cs_classify_intent)
    g.add_node("retrieve_context", _cs_retrieve_context)
    g.add_node("draft_reply", _cs_draft_reply)
    g.add_node("quality_check", _cs_quality_check)
    g.add_edge(START, "classify_intent")
    g.add_edge("classify_intent", "retrieve_context")
    g.add_edge("retrieve_context", "draft_reply")
    g.add_edge("draft_reply", "quality_check")
    g.add_edge("quality_check", END)
    return g.compile()


# # ---------------------------------------------------------------------------
# # Graph 2: code_review_agent
# # ---------------------------------------------------------------------------

# class ReviewState(TypedDict):
#     messages: Annotated[list, add_messages]
#     diff_summary: str
#     style_findings: str
#     security_findings: str
#     inject_error: str


# _RV_STYLE = _MockChatModel(
#     responses=[
#         "Style issues: (1) `processUser` → `process_user` (PEP 8). (2) Missing docstring on public method. (3) Magic number `86400` → `SECONDS_PER_DAY`.",
#         "No style issues. Code is clean, well-named, and follows project conventions.",
#         "Minor: line 34 exceeds 88 chars. Split the chained method call across lines.",
#         "Unused import `os` on line 3. Rename `tmp` to a descriptive variable.",
#     ],
#     model_id="gpt-4o",
#     mean_input_tokens=1100,
#     mean_output_tokens=320,
# )

# _RV_SECURITY = _MockChatModel(
#     responses=[
#         "No security issues found. Input validation present; parameterised queries used throughout.",
#         "HIGH: line 23 — raw string interpolation into SQL. Replace with parameterised query.\nLOW: line 67 — verbose error exposes stack trace.",
#         "MEDIUM: JWT secret read from env without validation — app crashes silently if var absent.\nLOW: CORS wildcard origin.",
#         "CRITICAL: line 45 — unsanitised user input passed to `subprocess.run()`. Command injection risk.",
#     ],
#     model_id="gpt-4o",
#     mean_input_tokens=1400,
#     mean_output_tokens=260,
# )

# _RV_SUMMARISER = _MockChatModel(
#     responses=[
#         "**Verdict: Request Changes**\n\nSQL injection risk on line 23 (HIGH) and missing input sanitisation on the registration endpoint. Style is otherwise clean.",
#         "**Verdict: Approved**\n\nCode quality is high. No security concerns. Pool refactor improves resilience. LGTM.",
#         "**Verdict: Request Changes**\n\nJWT middleware has a critical startup risk. Recommend a startup assertion and a configurable algorithm.",
#         "**Verdict: Reject**\n\nCRITICAL command injection on line 45. Must be fixed before any further review.",
#     ],
#     model_id="gpt-4o",
#     mean_input_tokens=1800,
#     mean_output_tokens=480,
# )

# _RV_INPUTS = [
#     ("Added input validation to user registration endpoint", "auth/registration.py"),
#     ("Refactored database connection pooling logic", "db/pool.py"),
#     ("Implemented rate limiting middleware", "middleware/rate_limiter.py"),
#     ("Added JWT authentication middleware", "auth/jwt_middleware.py"),
#     ("Optimised bulk insert performance for events table", "db/events_repo.py"),
# ]


# def _rv_parse_diff(state: ReviewState) -> dict:
#     if state.get("inject_error") == "parse_diff":
#         raise RuntimeError("Simulated failure: diff parser crashed on binary file")
#     human_msg = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
#     return {"diff_summary": human_msg[:150]}


# def _rv_analyse_style(state: ReviewState, config: RunnableConfig) -> dict:
#     if state.get("inject_error") == "analyse_style":
#         raise RuntimeError("Simulated failure: style analyser timeout")
#     file_hint = state.get("diff_summary", "").split()[-1] if state.get("diff_summary") else "main.py"
#     linter_result = run_linter.invoke({"file": file_hint}, config=config)
#     reply = _RV_STYLE.invoke(
#         [
#             SystemMessage(content="Analyse the diff for style issues, naming, and code organisation."),
#             HumanMessage(content=f"Diff: {state['diff_summary']}\nLinter output: {linter_result}"),
#         ],
#         config=config,
#     )
#     return {"style_findings": reply.content, "messages": [reply]}


# def _rv_check_security(state: ReviewState, config: RunnableConfig) -> dict:
#     if state.get("inject_error") == "check_security":
#         raise RuntimeError("Simulated failure: security scanner API key expired")
#     docs = search_docs.invoke({"query": state.get("diff_summary", "")[:80]}, config=config)
#     reply = _RV_SECURITY.invoke(
#         [
#             SystemMessage(content="Inspect the diff for vulnerabilities. Include severity (low/medium/high/critical)."),
#             HumanMessage(content=f"Diff: {state['diff_summary']}\nRelevant docs: {docs}"),
#         ],
#         config=config,
#     )
#     return {"security_findings": reply.content, "messages": [reply]}


# def _rv_summarise(state: ReviewState, config: RunnableConfig) -> dict:
#     if state.get("inject_error") == "summarise":
#         raise RuntimeError("Simulated failure: summariser context window exceeded")
#     reply = _RV_SUMMARISER.invoke(
#         [
#             SystemMessage(content="Synthesise code-review findings into a summary with an overall verdict."),
#             HumanMessage(content=f"Style: {state.get('style_findings', '')}\nSecurity: {state.get('security_findings', '')}"),
#         ],
#         config=config,
#     )
#     return {"messages": [reply]}


# def build_code_review_graph():
#     g = StateGraph(ReviewState)
#     g.add_node("parse_diff", _rv_parse_diff)
#     g.add_node("analyse_style", _rv_analyse_style)
#     g.add_node("check_security", _rv_check_security)
#     g.add_node("summarise", _rv_summarise)
#     g.add_edge(START, "parse_diff")
#     g.add_edge("parse_diff", "analyse_style")
#     g.add_edge("analyse_style", "check_security")
#     g.add_edge("check_security", "summarise")
#     g.add_edge("summarise", END)
#     return g.compile()


# # ---------------------------------------------------------------------------
# # Graph 3: research_agent
# # ---------------------------------------------------------------------------

# class ResearchState(TypedDict):
#     messages: Annotated[list, add_messages]
#     queries: str
#     search_results: str
#     synthesis: str
#     inject_error: str


# _RS_PLANNER = _MockChatModel(
#     responses=[
#         "1. 'RAG retrieval augmented generation production 2025'\n2. 'hybrid dense sparse retrieval BEIR'\n3. 'self-RAG adaptive retrieval'\n4. 'RAG evaluation hallucination reduction'",
#         "1. 'LLM hallucination mitigation 2024 2025'\n2. 'factuality RLHF training'\n3. 'chain of verification CoVe'\n4. 'RAG citation grounding factual accuracy'",
#         "1. 'code generation LLM HumanEval 2025'\n2. 'speculative decoding LLM inference'\n3. 'self-repair LLM code execution'\n4. 'retrieval augmented code generation'",
#     ],
#     model_id="claude-3-5-haiku-20241022",
#     mean_input_tokens=420,
#     mean_output_tokens=160,
# )

# _RS_SYNTHESISER = _MockChatModel(
#     responses=[
#         "Hybrid retrieval (dense + sparse) outperforms single-approach baselines by 8–12% on BEIR. Self-RAG achieves the best accuracy with lowest token cost. Re-ranking reduces hallucination by ~15%.",
#         "Multi-stage pipelines combining RLHF, RAG grounding, and self-verification are the current SotA. GPT-4o at 91.2% factuality on HalluBench 2025. CoVe adds ~20% latency but gains 12% factuality.",
#         "HumanEval+: Claude 3.7=94.1%, GPT-4o=93.8%, Gemini 2.0=91.4%. Speculative decoding reduces p50 latency by 40%. Test-based self-repair improves pass@1 by 6–8%.",
#     ],
#     model_id="claude-3-5-haiku-20241022",
#     mean_input_tokens=2800,
#     mean_output_tokens=560,
# )

# _RS_FORMATTER = _MockChatModel(
#     responses=[
#         "# RAG in Production (2025)\n\n## Executive Summary\nHybrid retrieval and re-ranking deliver 8–15% accuracy improvements.\n\n## Key Findings\n- Hybrid retrieval outperforms dense-only by 8–12% on BEIR\n- Re-ranking reduces hallucination by ~15%\n- Self-RAG achieves best accuracy at lowest cost\n\n## References\n1. arXiv:2310.11511 — Self-RAG\n2. 'RAG in Production' (arXiv, March 2025)",
#         "# LLM Hallucination Mitigation\n\n## Executive Summary\nMulti-stage pipelines are the current SotA for factual accuracy.\n\n## Key Findings\n- GPT-4o achieves 91.2% factuality on HalluBench 2025\n- Citation grounding cuts hallucination by up to 40%\n- CoVe adds ~20% latency but improves factuality by 12%\n\n## References\n1. HalluBench 2025\n2. Constitutional AI (arXiv:2212.08073)",
#         "# Code Generation with LLMs (2025)\n\n## Executive Summary\nTop models exceed 94% pass@1 on HumanEval+.\n\n## Key Findings\n- Claude 3.7 Sonnet leads at 94.1% pass@1\n- Speculative decoding cuts p50 latency by 40%\n- Test-based self-repair adds 6–8% pass@1\n\n## References\n1. HumanEval+ 2025 leaderboard\n2. arXiv:2402.01030 — RAPTOR",
#     ],
#     model_id="claude-3-5-haiku-20241022",
#     mean_input_tokens=1600,
#     mean_output_tokens=720,
# )

# _RS_INPUTS = [
#     "What are the latest production-ready advances in RAG systems?",
#     "Summarise recent work on LLM hallucination mitigation techniques.",
#     "What is the current state of the art for code generation with LLMs?",
# ]


# def _rs_plan_queries(state: ResearchState, config: RunnableConfig) -> dict:
#     if state.get("inject_error") == "plan_queries":
#         raise RuntimeError("Simulated failure: planner rejected input as too vague")
#     human_msg = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
#     reply = _RS_PLANNER.invoke(
#         [
#             SystemMessage(content="Decompose the research question into 3–5 targeted search queries."),
#             HumanMessage(content=f"Research question: {human_msg}"),
#         ],
#         config=config,
#     )
#     return {"queries": reply.content, "messages": [reply]}


# def _rs_web_search(state: ResearchState, config: RunnableConfig) -> dict:
#     if state.get("inject_error") == "web_search":
#         raise RuntimeError("Simulated failure: web search API rate limit exceeded")
#     first_query = (state.get("queries") or "").split("\n")[0].lstrip("1. ") or "LLM research 2025"
#     web = web_search_tool.invoke({"query": first_query[:80]}, config=config)
#     paper = arxiv_lookup.invoke({"title": first_query[:60]}, config=config)
#     return {"search_results": f"{web}\n\nArXiv: {paper}"}


# def _rs_synthesise(state: ResearchState, config: RunnableConfig) -> dict:
#     if state.get("inject_error") == "synthesise":
#         raise RuntimeError("Simulated failure: synthesiser context window exceeded")
#     human_msg = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
#     reply = _RS_SYNTHESISER.invoke(
#         [
#             SystemMessage(content="Synthesise search results into a coherent, structured answer. Cite sources."),
#             HumanMessage(content=f"Question: {human_msg}\n\nResults:\n{state.get('search_results', '')}"),
#         ],
#         config=config,
#     )
#     return {"synthesis": reply.content, "messages": [reply]}


# def _rs_format_report(state: ResearchState, config: RunnableConfig) -> dict:
#     if state.get("inject_error") == "format_report":
#         raise RuntimeError("Simulated failure: formatter crashed on markdown rendering")
#     reply = _RS_FORMATTER.invoke(
#         [
#             SystemMessage(content="Format the synthesised research into a well-structured markdown report."),
#             HumanMessage(content=state.get("synthesis", "")),
#         ],
#         config=config,
#     )
#     return {"messages": [reply]}


# def build_research_graph():
#     g = StateGraph(ResearchState)
#     g.add_node("plan_queries", _rs_plan_queries)
#     g.add_node("web_search", _rs_web_search)
#     g.add_node("synthesise", _rs_synthesise)
#     g.add_node("format_report", _rs_format_report)
#     g.add_edge(START, "plan_queries")
#     g.add_edge("plan_queries", "web_search")
#     g.add_edge("web_search", "synthesise")
#     g.add_edge("synthesise", "format_report")
#     g.add_edge("format_report", END)
#     return g.compile()


# ---------------------------------------------------------------------------
# Seed runner
# ---------------------------------------------------------------------------

def _maybe_fail(nodes: list[str]) -> str:
    """Return a node name to inject failure into (12% of runs), or ''."""
    if random.random() < 0.12:
        return random.choice(nodes)
    return ""


GRAPH_SPECS = [
    {
        "graph_id": "customer_support_agent",
        "build_fn": build_customer_support_graph,
        "fail_nodes": ["classify_intent", "retrieve_context", "draft_reply", "quality_check"],
        "make_inputs": lambda: {
            "messages": [HumanMessage(content=random.choice(_CS_INPUTS))],
            "context": "",
            "intent": "",
            "inject_error": "",
        },
    },
    # {
    #     "graph_id": "code_review_agent",
    #     "build_fn": build_code_review_graph,
    #     "fail_nodes": ["parse_diff", "analyse_style", "check_security", "summarise"],
    #     "make_inputs": lambda: {
    #         "messages": [HumanMessage(content=random.choice(_RV_INPUTS)[0])],
    #         "diff_summary": "",
    #         "style_findings": "",
    #         "security_findings": "",
    #         "inject_error": "",
    #     },
    # },
    # {
    #     "graph_id": "research_agent",
    #     "build_fn": build_research_graph,
    #     "fail_nodes": ["plan_queries", "web_search", "synthesise", "format_report"],
    #     "make_inputs": lambda: {
    #         "messages": [HumanMessage(content=random.choice(_RS_INPUTS))],
    #         "queries": "",
    #         "search_results": "",
    #         "synthesis": "",
    #         "inject_error": "",
    #     },
    # },
]


def seed(num_runs: int = 80) -> tuple[int, int]:
    graphs = {spec["graph_id"]: spec["build_fn"]() for spec in GRAPH_SPECS}

    completed = 0
    failed = 0

    for i in range(num_runs):
        spec = GRAPH_SPECS[i % len(GRAPH_SPECS)]
        graph_id = spec["graph_id"]
        thread_id = random.choice(THREADS)

        inputs = spec["make_inputs"]()
        inputs["inject_error"] = _maybe_fail(spec["fail_nodes"])

        monitor = LangGraphMonitorCallback(graph_id=graph_id, thread_id=thread_id, pricing=PRICING)
        try:
            graphs[graph_id].invoke(inputs, config={"callbacks": [monitor]})
            completed += 1
        except Exception:
            failed += 1

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{num_runs} runs  (completed={completed}, failed={failed})")

    return completed, failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed MongoDB with realistic LangGraph runs.")
    parser.add_argument("--runs", type=int, default=80, help="Total number of runs to execute (default: 80)")
    args = parser.parse_args()

    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    db.runs.drop()
    db.events.drop()
    print(f"Cleared '{DB_NAME}.runs' and '{DB_NAME}.events'.")
    print(f"Seeding {args.runs} runs across {len(GRAPH_SPECS)} graphs...")

    completed, failed = seed(args.runs)

    total_runs = db.runs.count_documents({})
    total_events = db.events.count_documents({})
    agg = next(
        db.runs.aggregate([
            {"$group": {
                "_id": None,
                "cost": {"$sum": "$estimated_cost_usd"},
                "in_tok": {"$sum": "$total_input_tokens"},
                "out_tok": {"$sum": "$total_output_tokens"},
            }}
        ]),
        {},
    )

    print(f"\nDone.")
    print(f"  runs inserted    : {total_runs}")
    print(f"  events inserted  : {total_events}")
    print(f"  completed        : {completed}")
    print(f"  failed           : {failed}")
    print(f"  total input tok  : {agg.get('in_tok', 0):,}")
    print(f"  total output tok : {agg.get('out_tok', 0):,}")
    print(f"  estimated cost   : ${agg.get('cost', 0):.4f} USD")


if __name__ == "__main__":
    main()
