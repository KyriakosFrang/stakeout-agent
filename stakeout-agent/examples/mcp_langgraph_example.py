"""Example: LangGraph agent that uses the stakeout MCP server to detect and avoid
repeating previously failed strategies.

Prerequisites
-------------
1.  Start MongoDB and seed some failed runs:
        docker compose up -d mongo
        python examples/seed_demo_data.py

2.  Start the stakeout MCP server in a separate terminal:
        MONGO_URI=mongodb://localhost:27017 stakeout mcp --transport sse --port 8001

3.  Install dependencies:
        pip install 'stakeout-agent[mcp,langgraph,mongodb]' langchain-mcp-adapters anthropic

4.  Run this script:
        ANTHROPIC_API_KEY=<your-key> python examples/mcp_langgraph_example.py
"""

import sys

sys.path.insert(0, "..")  # resolve stakeout_agent from the repo root when running directly

import asyncio

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph, MessagesState

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
except ImportError:
    print("langchain-mcp-adapters is required. Run: pip install langchain-mcp-adapters")
    sys.exit(1)

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    print("langchain-anthropic is required. Run: pip install langchain-anthropic")
    sys.exit(1)


GRAPH_ID = "research-agent"
MCP_URL = "http://127.0.0.1:8001/sse"


async def build_agent():
    client = MultiServerMCPClient({"stakeout": {"url": MCP_URL, "transport": "sse"}})
    tools = await client.get_tools()

    llm = ChatAnthropic(model="claude-sonnet-4-6").bind_tools(tools)

    def call_model(state: MessagesState):
        return {"messages": [llm.invoke(state["messages"])]}

    def call_tool(state: MessagesState):
        from langchain_core.messages import ToolMessage

        last = state["messages"][-1]
        results = []
        for call in last.tool_calls:
            tool = next(t for t in tools if t.name == call["name"])
            output = asyncio.get_event_loop().run_until_complete(tool.ainvoke(call["args"]))
            results.append(ToolMessage(content=str(output), tool_call_id=call["id"]))
        return {"messages": results}

    def should_continue(state: MessagesState):
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return END

    graph = StateGraph(MessagesState)
    graph.add_node("model", call_model)
    graph.add_node("tools", call_tool)
    graph.set_entry_point("model")
    graph.add_conditional_edges("model", should_continue)
    graph.add_edge("tools", "model")
    return graph.compile()


async def main():
    agent = await build_agent()

    prompt = (
        f"I'm about to run the '{GRAPH_ID}' agent to fetch the latest news. "
        f"Before I start, check if there have been any failed runs for this graph "
        f"in the last 2 hours using the stakeout MCP tools. "
        f"If there are repeated failures at the same node, suggest an alternative approach. "
        f"Then report the overall stats for the last 7 days."
    )

    print("User:", prompt)
    print()
    result = await agent.ainvoke({"messages": [HumanMessage(content=prompt)]})
    final = result["messages"][-1]
    print("Agent:", final.content)


if __name__ == "__main__":
    asyncio.run(main())
