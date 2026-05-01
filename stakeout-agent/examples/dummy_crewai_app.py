"""
Dummy CrewAI application to manually verify stakeout-agent monitoring (sync).

Run MongoDB first:
    docker compose up -d mongo

Then run this script:
    uv run python examples/dummy_crewai_app.py

It will print every document written to the `runs` and `events` collections.
"""

import json
import logging

from crewai import Agent, Crew, Task
from crewai.tools import BaseTool

from stakeout_agent import CrewAIMonitorCallback
from stakeout_agent.backends.mongodb import MongoMonitorDB

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s — %(message)s")


# ---------------------------------------------------------------------------
# A trivial tool (no LLM required)
# ---------------------------------------------------------------------------

class MultiplyTool(BaseTool):
    name: str = "multiply"
    description: str = "Multiply two integers. Input must be a string like '4 3'."

    def _run(self, input: str) -> str:
        a, b = (int(x) for x in input.strip().split())
        return str(a * b)


# ---------------------------------------------------------------------------
# Agents and tasks
# ---------------------------------------------------------------------------

def build_crew() -> Crew:
    multiply_tool = MultiplyTool()

    researcher = Agent(
        role="Researcher",
        goal="Double the given number",
        backstory="You are a precise mathematician.",
        tools=[multiply_tool],
        verbose=True,
    )

    writer = Agent(
        role="Writer",
        goal="Summarise the computation result",
        backstory="You write clear summaries.",
        verbose=True,
    )

    double_task = Task(
        description="Multiply 5 by 2 using the multiply tool and report the result.",
        expected_output="The result of 5 * 2.",
        agent=researcher,
    )

    summary_task = Task(
        description="Summarise the result from the previous task in one sentence.",
        expected_output="A one-sentence summary.",
        agent=writer,
    )

    return Crew(agents=[researcher, writer], tasks=[double_task, summary_task], verbose=True)


# ---------------------------------------------------------------------------
# Run and inspect
# ---------------------------------------------------------------------------

def main():
    crew = build_crew()

    monitor = CrewAIMonitorCallback(crew_id="dummy_crew", thread_id="thread_001")

    print("\n--- Running crew ---")
    result = crew.kickoff()
    print(f"\n[crew] final output: {result}")

    print("\n--- Inspecting MongoDB ---")
    db = MongoMonitorDB()

    run = db.runs.find_one({"graph_id": "dummy_crew"}, sort=[("started_at", -1)])
    if run is None:
        print("ERROR: no run document found — is MongoDB running?")
        return

    print(f"\nrun document:\n{json.dumps(run, default=str, indent=2)}")

    events = list(db.events.find({"run_id": run["_id"]}).sort("timestamp", 1))
    print(f"\nevents ({len(events)} total):")
    for ev in events:
        latency = f"{ev['latency_ms']} ms" if "latency_ms" in ev else "-"
        print(f"  [{ev['event_type']:12s}] node={ev['node_name']:15s} latency={latency}")


if __name__ == "__main__":
    main()
