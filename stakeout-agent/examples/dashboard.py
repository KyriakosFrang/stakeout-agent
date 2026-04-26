"""
Stakeout Agent — Example Dashboard

Usage:
    docker compose up -d mongo
    uv run python examples/seed_demo_data.py   # optional: load demo data
    uv run --with streamlit streamlit run examples/dashboard.py
"""

import os

import pandas as pd
import streamlit as st
from pymongo import MongoClient

# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

@st.cache_resource
def _get_db():
    uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    db_name = os.getenv("MONGO_DB", "stakeout")
    return MongoClient(uri)[db_name]


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

@st.cache_data(ttl=10)
def _load_runs() -> pd.DataFrame:
    docs = list(_get_db().runs.find().sort("started_at", -1).limit(500))
    if not docs:
        return pd.DataFrame()
    df = pd.DataFrame(docs).rename(columns={"_id": "run_id"})
    df["started_at"] = pd.to_datetime(df["started_at"], utc=True)
    df["ended_at"] = pd.to_datetime(df["ended_at"], utc=True)
    df["duration_ms"] = (df["ended_at"] - df["started_at"]).dt.total_seconds() * 1000
    return df


@st.cache_data(ttl=10)
def _load_events() -> pd.DataFrame:
    docs = list(_get_db().events.find().sort("timestamp", -1).limit(10_000))
    if not docs:
        return pd.DataFrame()
    df = pd.DataFrame(docs)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Stakeout — Graph Monitor", layout="wide")
st.title("Stakeout Agent — Graph Monitor")

runs_df = _load_runs()
events_df = _load_events()

if runs_df.empty:
    st.warning(
        "No runs found in MongoDB. "
        "Run `python examples/seed_demo_data.py` to load demo data, "
        "or start your LangGraph app with the monitor callback attached."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Filters")

    all_graphs = ["All"] + sorted(runs_df["graph_id"].unique().tolist())
    selected_graph = st.selectbox("Graph ID", all_graphs)

    selected_status = st.selectbox("Status", ["All", "completed", "failed", "running"])

    if st.button("Refresh data"):
        st.cache_data.clear()
        st.rerun()

filtered = runs_df.copy()
if selected_graph != "All":
    filtered = filtered[filtered["graph_id"] == selected_graph]
if selected_status != "All":
    filtered = filtered[filtered["status"] == selected_status]

ev_filtered = events_df.copy()
if selected_graph != "All" and "graph_id" in ev_filtered.columns:
    ev_filtered = ev_filtered[ev_filtered["graph_id"] == selected_graph]

# ---------------------------------------------------------------------------
# Top-line metrics
# ---------------------------------------------------------------------------

total = len(filtered)
completed = int((filtered["status"] == "completed").sum())
failed = int((filtered["status"] == "failed").sum())
running = int((filtered["status"] == "running").sum())
success_rate = completed / total * 100 if total else 0
avg_dur = filtered.loc[filtered["status"] == "completed", "duration_ms"].mean()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Runs", total)
c2.metric("Completed", completed)
c3.metric("Failed", failed)
c4.metric("Success Rate", f"{success_rate:.1f}%")
c5.metric("Avg Duration", f"{avg_dur:.0f} ms" if pd.notna(avg_dur) else "—")

st.divider()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_history, tab_nodes, tab_inspector, tab_thread = st.tabs(
    ["Run History", "Node Performance", "Run Inspector", "Conversation Deep Dive"]
)

# ---- Tab 1: Run History ------------------------------------------------

with tab_history:
    st.subheader("Recent Runs")
    display_cols = [c for c in ["run_id", "graph_id", "thread_id", "status", "started_at", "duration_ms", "error"] if c in filtered.columns]
    st.dataframe(
        filtered[display_cols].head(100),
        use_container_width=True,
        column_config={
            "duration_ms": st.column_config.NumberColumn("Duration (ms)", format="%.0f"),
            "started_at": st.column_config.DatetimeColumn("Started At", format="YYYY-MM-DD HH:mm:ss"),
        },
        hide_index=True,
    )

    if len(filtered) > 1:
        st.subheader("Runs Over Time")
        ts = (
            filtered.set_index("started_at")
            .resample("10min")["run_id"]
            .count()
            .rename("runs")
            .reset_index()
        )
        st.line_chart(ts.set_index("started_at"))

    if failed > 0:
        st.subheader("Recent Failures")
        fail_df = filtered[filtered["status"] == "failed"][["run_id", "graph_id", "started_at", "error"]].head(10)
        st.dataframe(fail_df, use_container_width=True, hide_index=True)

# ---- Tab 2: Node Performance -------------------------------------------

with tab_nodes:
    if ev_filtered.empty:
        st.info("No events recorded yet.")
    else:
        node_ends = ev_filtered[ev_filtered["event_type"] == "node_end"].dropna(subset=["latency_ms"])

        if node_ends.empty:
            st.info("No node_end events with latency data yet.")
        else:
            st.subheader("Node Latency")
            agg = (
                node_ends.groupby("node_name")["latency_ms"]
                .agg(avg="mean", p95=lambda x: x.quantile(0.95), calls="count")
                .sort_values("avg", ascending=False)
                .reset_index()
                .rename(columns={"node_name": "Node", "avg": "Avg (ms)", "p95": "P95 (ms)", "calls": "Calls"})
            )
            agg["Avg (ms)"] = agg["Avg (ms)"].round(1)
            agg["P95 (ms)"] = agg["P95 (ms)"].round(1)

            st.bar_chart(agg.set_index("Node")["Avg (ms)"])
            st.dataframe(agg, use_container_width=True, hide_index=True)

        tool_results = ev_filtered[ev_filtered["event_type"] == "tool_result"].dropna(subset=["latency_ms"])

        if not tool_results.empty:
            st.subheader("Tool Latency")
            tool_agg = (
                tool_results.groupby("node_name")["latency_ms"]
                .agg(avg="mean", calls="count")
                .sort_values("avg", ascending=False)
                .reset_index()
                .rename(columns={"node_name": "Tool", "avg": "Avg (ms)", "calls": "Calls"})
            )
            tool_agg["Avg (ms)"] = tool_agg["Avg (ms)"].round(1)
            st.bar_chart(tool_agg.set_index("Tool")["Avg (ms)"])
            st.dataframe(tool_agg, use_container_width=True, hide_index=True)

        error_events = ev_filtered[ev_filtered["event_type"] == "error"]
        if not error_events.empty:
            st.subheader("Errors by Node")
            err_agg = (
                error_events.groupby("node_name")
                .size()
                .reset_index(name="count")
                .sort_values("count", ascending=False)
            )
            st.bar_chart(err_agg.set_index("node_name"))

# ---- Tab 3: Run Inspector ----------------------------------------------

with tab_inspector:
    st.subheader("Inspect a Run")
    run_ids = filtered["run_id"].tolist()

    if not run_ids:
        st.info("No runs match the current filter.")
    else:
        selected_run_id = st.selectbox("Select Run ID", run_ids)
        run_row = filtered[filtered["run_id"] == selected_run_id].iloc[0]

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Status", run_row["status"])
        r2.metric("Graph", run_row["graph_id"])
        r3.metric("Thread", run_row["thread_id"])
        dur = run_row.get("duration_ms")
        r4.metric("Duration", f"{dur:.0f} ms" if pd.notna(dur) else "—")

        if pd.notna(run_row.get("error")):
            st.error(f"Error: {run_row['error']}")

        st.subheader("Event Timeline")
        run_events = (
            events_df[events_df["run_id"] == selected_run_id]
            .sort_values("timestamp")[["timestamp", "event_type", "node_name", "latency_ms", "error"]]
        )

        if run_events.empty:
            st.info("No events found for this run.")
        else:
            st.dataframe(
                run_events,
                use_container_width=True,
                column_config={
                    "latency_ms": st.column_config.NumberColumn("Latency (ms)", format="%.1f"),
                    "timestamp": st.column_config.DatetimeColumn("Timestamp", format="HH:mm:ss.SSS"),
                },
                hide_index=True,
            )

# ---- Tab 4: Run Deep Dive -------------------------------------------

with tab_thread:
    st.subheader("Conversation Run Deep Dive — Human vs AI")

    all_runs = sorted(runs_df["run_id"].unique().tolist())
    if not all_runs:
        st.info("No runs found.")
        st.stop()

    selected_run = st.selectbox("Select Run", all_runs, key="run_select")

    thread_runs = (
        runs_df[runs_df["run_id"] == selected_run]
        .sort_values("started_at")
        .to_dict("records")
    )

    # Thread-level stats
    n_runs = len(thread_runs)
    n_completed = sum(1 for r in thread_runs if r["status"] == "completed")
    n_failed = sum(1 for r in thread_runs if r["status"] == "failed")
    thread_graphs = list({r["graph_id"] for r in thread_runs})

    ts1, ts2, ts3, ts4 = st.columns(4)
    ts1.metric("Turns (runs)", n_runs)
    ts2.metric("Completed", n_completed)
    ts3.metric("Failed", n_failed)
    ts4.metric("Graph", thread_graphs[0] if len(thread_graphs) == 1 else "mixed")

    st.divider()

    # Conversation replay
    thread_run_ids = [r["run_id"] for r in thread_runs]
    thread_events_df = events_df[events_df["run_id"].isin(thread_run_ids)].sort_values("timestamp")

    if thread_events_df.empty:
        st.info("No events recorded for this thread.")
    else:
        for run in thread_runs:
            run_id = run["run_id"]
            run_evs = thread_events_df[thread_events_df["run_id"] == run_id]

            # Extract human message: top-level `messages` on the first node_start event.
            # base.py stores extracted LangChain messages there as plain {role, content} dicts.
            first_start = run_evs[run_evs["event_type"] == "node_start"].iloc[:1]
            human_msg = None
            if not first_start.empty:
                msgs = first_start.iloc[0].get("messages")
                if isinstance(msgs, list) and msgs:
                    last = msgs[-1]
                    if isinstance(last, dict) and last.get("role") == "human":
                        human_msg = last.get("content")

            # Extract AI message: top-level `messages` on the last node_end event.
            node_ends = run_evs[run_evs["event_type"] == "node_end"]
            ai_msg = None
            if not node_ends.empty:
                msgs = node_ends.iloc[-1].get("messages")
                if isinstance(msgs, list) and msgs:
                    last = msgs[-1]
                    if isinstance(last, dict) and last.get("role") == "assistant":
                        ai_msg = last.get("content")

            # Tool calls for this run
            tool_calls = run_evs[run_evs["event_type"] == "tool_call"]
            tool_results = run_evs[run_evs["event_type"] == "tool_result"]
            tool_result_map = {
                row["node_name"]: row for _, row in tool_results.iterrows()
            }

            # Render human turn
            if human_msg:
                with st.chat_message("human"):
                    st.write(human_msg)

            # Render tool calls inline (between human and AI)
            if not tool_calls.empty:
                for _, tc in tool_calls.iterrows():
                    with st.chat_message("tool", avatar="🔧"):
                        tr = tool_result_map.get(tc["node_name"])
                        latency_label = (
                            f"{tr['latency_ms']:.0f} ms" if tr is not None and pd.notna(tr.get("latency_ms")) else "—"
                        )
                        with st.expander(f"`{tc['node_name']}` — {latency_label}"):
                            tc_payload = tc.get("payload", {})
                            tc_input = tc_payload.get("input", "") if isinstance(tc_payload, dict) else ""
                            st.markdown(f"**Input:** {tc_input}")
                            if tr is not None:
                                tr_payload = tr.get("payload", {})
                                tr_output = tr_payload.get("output", "") if isinstance(tr_payload, dict) else ""
                                st.markdown(f"**Output:** {tr_output}")

            # Render AI response or error
            if run["status"] == "failed":
                with st.chat_message("assistant"):
                    st.error(f"Run failed: {run.get('error', 'unknown error')}")
                    dur = run.get("duration_ms")
                    st.caption(
                        f"Run `{run_id[:8]}…` · failed · "
                        f"{dur:.0f} ms" if pd.notna(dur) else f"Run `{run_id[:8]}…` · failed"
                    )
            elif ai_msg:
                with st.chat_message("assistant"):
                    st.write(ai_msg)
                    dur = run.get("duration_ms")
                    st.caption(
                        f"Run `{run_id[:8]}…` · completed · "
                        + (f"{dur:.0f} ms" if pd.notna(dur) else "")
                    )
            elif not human_msg:
                # No message content available — fall back to event table for this run
                with st.expander(f"Run `{run_id[:8]}…` (no message content)"):
                    st.dataframe(
                        run_evs[["timestamp", "event_type", "node_name", "latency_ms"]],
                        use_container_width=True,
                        hide_index=True,
                    )

            st.divider()
