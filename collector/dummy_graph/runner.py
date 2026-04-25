"""
Continuously runs the dummy graph to generate monitoring data.
Simulates realistic usage with varied queries and random delays.
"""
import random
import time
import uuid
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

from dummy_graph.graph import build_graph
from stakeout_agent import LangGraphMonitorCallback, MonitorDB

SAMPLE_QUERIES = [
    "monthly revenue by product line",
    "customer churn rate Q4",
    "loan portfolio risk exposure",
    "transaction anomaly detection",
    "branch performance metrics",
    "credit score distribution",
    "FX exposure by currency",
    "operational cost analysis",
    "NPS score by segment",
    "liquidity ratio forecast",
]

GRAPH_IDS = ["risk-agent", "analytics-agent", "reporting-agent"]


def run_once(graph, db: MonitorDB) -> None:
    graph_id = random.choice(GRAPH_IDS)
    thread_id = str(uuid.uuid4())
    query = random.choice(SAMPLE_QUERIES)

    monitor = LangGraphMonitorCallback(
        graph_id=graph_id,
        thread_id=thread_id,
        db=db,
    )

    log.info(f"Running graph={graph_id} thread={thread_id} query='{query}'")

    try:
        result = graph.invoke(
            {"query": query, "plan": "", "result": "", "summary": ""},
            config={"callbacks": [monitor]},
        )
        log.info(f"Completed: {result.get('summary', '')[:80]}")
    except Exception as e:
        log.error(f"Graph failed: {e}")


def main():
    log.info("Starting graph runner...")
    db = MonitorDB()
    graph = build_graph()

    while True:
        try:
            run_once(graph, db)
        except Exception as e:
            log.error(f"Runner error: {e}")

        delay = random.uniform(3, 8)
        log.info(f"Next run in {delay:.1f}s")
        time.sleep(delay)


if __name__ == "__main__":
    main()
