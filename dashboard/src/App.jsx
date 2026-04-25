import { useState, useEffect, useCallback } from 'react'

const API = '/api'

const css = `
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg: #080b0f;
    --bg-panel: #0d1117;
    --bg-card: #111820;
    --bg-hover: #161f2a;
    --border: #1e2d3d;
    --border-bright: #2a3f55;
    --text: #c9d8e8;
    --text-dim: #556a7e;
    --text-bright: #e8f2ff;
    --green: #00d68f;
    --green-dim: #00855a;
    --red: #ff4757;
    --red-dim: #7a1a22;
    --amber: #ffb300;
    --amber-dim: #7a5500;
    --blue: #2196f3;
    --blue-dim: #0d3a6b;
    --mono: 'IBM Plex Mono', monospace;
    --sans: 'IBM Plex Sans', sans-serif;
  }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    font-size: 14px;
    min-height: 100vh;
  }

  /* Layout */
  .shell { display: flex; flex-direction: column; min-height: 100vh; }

  .topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 24px;
    height: 52px;
    background: var(--bg-panel);
    border-bottom: 1px solid var(--border);
    position: sticky; top: 0; z-index: 100;
  }

  .topbar-brand {
    display: flex; align-items: center; gap: 10px;
    font-family: var(--mono); font-size: 13px; font-weight: 600;
    color: var(--text-bright); letter-spacing: 0.08em;
  }

  .brand-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--green);
    box-shadow: 0 0 6px var(--green);
    animation: pulse 2s infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; } 50% { opacity: 0.4; }
  }

  .topbar-right {
    display: flex; align-items: center; gap: 16px;
    font-family: var(--mono); font-size: 11px; color: var(--text-dim);
  }

  .refresh-btn {
    background: none; border: 1px solid var(--border-bright);
    color: var(--text-dim); cursor: pointer; border-radius: 4px;
    padding: 4px 10px; font-family: var(--mono); font-size: 11px;
    transition: all 0.15s;
  }
  .refresh-btn:hover { color: var(--text-bright); border-color: var(--blue); }

  .main { display: flex; flex: 1; overflow: hidden; }

  /* Stats bar */
  .stats-bar {
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 1px; background: var(--border);
    border-bottom: 1px solid var(--border);
  }

  .stat-cell {
    background: var(--bg-panel);
    padding: 14px 24px;
  }

  .stat-label {
    font-family: var(--mono); font-size: 10px; font-weight: 500;
    color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.12em;
    margin-bottom: 4px;
  }

  .stat-value {
    font-family: var(--mono); font-size: 22px; font-weight: 600;
    color: var(--text-bright);
  }

  .stat-value.green { color: var(--green); }
  .stat-value.red { color: var(--red); }
  .stat-value.amber { color: var(--amber); }

  /* Content layout */
  .content { display: flex; flex: 1; overflow: hidden; }

  .panel-left {
    width: 420px; flex-shrink: 0;
    border-right: 1px solid var(--border);
    display: flex; flex-direction: column;
    overflow: hidden;
  }

  .panel-right {
    flex: 1; overflow-y: auto;
    padding: 20px 24px;
  }

  .panel-header {
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
    font-family: var(--mono); font-size: 10px; font-weight: 600;
    color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.12em;
    display: flex; align-items: center; justify-content: space-between;
    background: var(--bg-panel);
  }

  .filter-row {
    padding: 8px 12px;
    border-bottom: 1px solid var(--border);
    background: var(--bg-panel);
    display: flex; gap: 6px;
  }

  .filter-btn {
    background: none; border: 1px solid var(--border);
    color: var(--text-dim); cursor: pointer; border-radius: 3px;
    padding: 3px 8px; font-family: var(--mono); font-size: 10px;
    transition: all 0.12s;
  }
  .filter-btn:hover { border-color: var(--border-bright); color: var(--text); }
  .filter-btn.active { background: var(--blue-dim); border-color: var(--blue); color: var(--blue); }

  /* Runs list */
  .runs-list { overflow-y: auto; flex: 1; }

  .run-item {
    padding: 10px 16px;
    border-bottom: 1px solid var(--border);
    cursor: pointer; transition: background 0.1s;
    display: flex; flex-direction: column; gap: 4px;
  }
  .run-item:hover { background: var(--bg-hover); }
  .run-item.selected { background: var(--bg-card); border-left: 2px solid var(--blue); }

  .run-item-top {
    display: flex; align-items: center; justify-content: space-between;
  }

  .run-graph-id {
    font-family: var(--mono); font-size: 11px; font-weight: 600;
    color: var(--text-bright);
  }

  .status-badge {
    font-family: var(--mono); font-size: 9px; font-weight: 600;
    padding: 2px 6px; border-radius: 2px; text-transform: uppercase; letter-spacing: 0.08em;
  }
  .status-badge.completed { background: var(--green-dim); color: var(--green); }
  .status-badge.failed { background: var(--red-dim); color: var(--red); }
  .status-badge.running { background: var(--blue-dim); color: var(--blue); }

  .run-item-meta {
    display: flex; gap: 12px;
    font-family: var(--mono); font-size: 10px; color: var(--text-dim);
  }

  /* Run detail */
  .detail-empty {
    display: flex; align-items: center; justify-content: center;
    height: 100%; color: var(--text-dim);
    font-family: var(--mono); font-size: 12px;
  }

  .detail-header { margin-bottom: 20px; }

  .detail-title {
    font-family: var(--mono); font-size: 13px; font-weight: 600;
    color: var(--text-bright); margin-bottom: 8px;
  }

  .detail-meta-grid {
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: 8px; margin-bottom: 20px;
  }

  .meta-card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 4px; padding: 10px 12px;
  }

  .meta-card-label {
    font-family: var(--mono); font-size: 9px; text-transform: uppercase;
    letter-spacing: 0.1em; color: var(--text-dim); margin-bottom: 4px;
  }

  .meta-card-value {
    font-family: var(--mono); font-size: 12px; color: var(--text-bright);
  }

  /* Event timeline */
  .section-title {
    font-family: var(--mono); font-size: 10px; font-weight: 600;
    color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.12em;
    margin-bottom: 12px;
  }

  .timeline { display: flex; flex-direction: column; gap: 2px; }

  .event-row {
    display: grid;
    grid-template-columns: 28px 90px 140px 80px 1fr;
    align-items: center; gap: 8px;
    padding: 7px 10px;
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 3px; font-family: var(--mono); font-size: 11px;
    transition: background 0.1s;
  }
  .event-row:hover { background: var(--bg-hover); }
  .event-row.error-row { border-color: var(--red-dim); background: #120608; }

  .event-icon { font-size: 14px; text-align: center; }

  .event-type {
    font-size: 9px; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600;
  }
  .event-type.node_start { color: var(--blue); }
  .event-type.node_end { color: var(--green); }
  .event-type.tool_call { color: var(--amber); }
  .event-type.tool_result { color: var(--amber); }
  .event-type.error { color: var(--red); }

  .event-node { color: var(--text); font-size: 11px; }

  .event-latency { color: var(--text-dim); font-size: 10px; text-align: right; }

  .event-error { color: var(--red); font-size: 10px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .event-payload { color: var(--text-dim); font-size: 10px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

  /* Loading / empty */
  .loading {
    padding: 24px; text-align: center;
    font-family: var(--mono); font-size: 11px; color: var(--text-dim);
  }

  .error-msg {
    padding: 12px 16px; background: #120608; border: 1px solid var(--red-dim);
    border-radius: 4px; font-family: var(--mono); font-size: 11px; color: var(--red);
    margin-bottom: 16px;
  }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--border-bright); border-radius: 2px; }
`

// ── Helpers ───────────────────────────────────────────────────────────────────

const fmt = {
  time: (iso) => {
    if (!iso) return '—'
    const d = new Date(iso)
    return d.toLocaleTimeString('en-GB', { hour12: false })
  },
  date: (iso) => {
    if (!iso) return '—'
    return new Date(iso).toLocaleString('en-GB', { hour12: false })
  },
  ms: (ms) => ms != null ? `${ms.toFixed(0)}ms` : '—',
  duration: (start, end) => {
    if (!start || !end) return '—'
    const ms = new Date(end) - new Date(start)
    return ms > 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`
  },
  shortId: (id) => id ? id.slice(0, 8) : '—',
}

const EVENT_ICONS = {
  node_start: '▶',
  node_end: '✓',
  tool_call: '⚡',
  tool_result: '↩',
  error: '✗',
}

const STATUS_LABELS = ['all', 'completed', 'failed', 'running']

// ── Components ────────────────────────────────────────────────────────────────

function StatBar({ stats }) {
  if (!stats) return null
  return (
    <div className="stats-bar">
      <div className="stat-cell">
        <div className="stat-label">Total Runs</div>
        <div className="stat-value">{stats.total_runs}</div>
      </div>
      <div className="stat-cell">
        <div className="stat-label">Success Rate</div>
        <div className={`stat-value ${stats.success_rate >= 90 ? 'green' : stats.success_rate >= 70 ? 'amber' : 'red'}`}>
          {stats.success_rate}%
        </div>
      </div>
      <div className="stat-cell">
        <div className="stat-label">Failed</div>
        <div className={`stat-value ${stats.failed > 0 ? 'red' : ''}`}>{stats.failed}</div>
      </div>
      <div className="stat-cell">
        <div className="stat-label">Running</div>
        <div className={`stat-value ${stats.running > 0 ? 'amber' : ''}`}>{stats.running}</div>
      </div>
    </div>
  )
}

function RunItem({ run, selected, onClick }) {
  return (
    <div className={`run-item ${selected ? 'selected' : ''}`} onClick={onClick}>
      <div className="run-item-top">
        <span className="run-graph-id">{run.graph_id}</span>
        <span className={`status-badge ${run.status}`}>{run.status}</span>
      </div>
      <div className="run-item-meta">
        <span>{fmt.shortId(run.id)}</span>
        <span>{fmt.time(run.started_at)}</span>
        <span>{fmt.duration(run.started_at, run.ended_at)}</span>
      </div>
    </div>
  )
}

function EventRow({ event }) {
  const isError = event.event_type === 'error'
  const detail = isError
    ? event.error
    : event.payload?.outputs
      ? JSON.stringify(event.payload.outputs).slice(0, 60)
      : event.payload?.input || ''

  return (
    <div className={`event-row ${isError ? 'error-row' : ''}`}>
      <span className="event-icon">{EVENT_ICONS[event.event_type] || '·'}</span>
      <span className={`event-type ${event.event_type}`}>{event.event_type.replace('_', ' ')}</span>
      <span className="event-node">{event.node_name}</span>
      <span className="event-latency">{fmt.ms(event.latency_ms)}</span>
      <span className={isError ? 'event-error' : 'event-payload'}>{detail}</span>
    </div>
  )
}

function RunDetail({ runId }) {
  const [run, setRun] = useState(null)
  const [events, setEvents] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!runId) return
    setLoading(true)
    setError(null)

    Promise.all([
      fetch(`${API}/runs/${runId}`).then(r => r.json()),
      fetch(`${API}/runs/${runId}/events`).then(r => r.json()),
    ])
      .then(([r, e]) => { setRun(r); setEvents(e) })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [runId])

  if (!runId) return <div className="detail-empty">← select a run to inspect</div>
  if (loading) return <div className="loading">loading run...</div>
  if (error) return <div className="error-msg">Error: {error}</div>

  const nodeEnds = events.filter(e => e.event_type === 'node_end' && e.latency_ms)
  const totalLatency = nodeEnds.reduce((s, e) => s + e.latency_ms, 0)
  const errorCount = events.filter(e => e.event_type === 'error').length

  return (
    <div>
      <div className="detail-header">
        <div className="detail-title">{run.graph_id} / {fmt.shortId(run.id)}</div>
      </div>

      <div className="detail-meta-grid">
        <div className="meta-card">
          <div className="meta-card-label">Status</div>
          <div className="meta-card-value"><span className={`status-badge ${run.status}`}>{run.status}</span></div>
        </div>
        <div className="meta-card">
          <div className="meta-card-label">Duration</div>
          <div className="meta-card-value">{fmt.duration(run.started_at, run.ended_at)}</div>
        </div>
        <div className="meta-card">
          <div className="meta-card-label">Node Latency</div>
          <div className="meta-card-value">{fmt.ms(totalLatency || null)}</div>
        </div>
        <div className="meta-card">
          <div className="meta-card-label">Started</div>
          <div className="meta-card-value">{fmt.date(run.started_at)}</div>
        </div>
        <div className="meta-card">
          <div className="meta-card-label">Events</div>
          <div className="meta-card-value">{events.length}</div>
        </div>
        <div className="meta-card">
          <div className="meta-card-label">Errors</div>
          <div className="meta-card-value" style={{ color: errorCount > 0 ? 'var(--red)' : 'inherit' }}>{errorCount}</div>
        </div>
      </div>

      {run.error && <div className="error-msg">{run.error}</div>}

      <div className="section-title">Event Timeline</div>
      <div className="timeline">
        {events.length === 0
          ? <div className="loading">no events recorded</div>
          : events.map((e, i) => <EventRow key={i} event={e} />)
        }
      </div>
    </div>
  )
}

// ── App ───────────────────────────────────────────────────────────────────────

export default function App() {
  const [runs, setRuns] = useState([])
  const [stats, setStats] = useState(null)
  const [selectedRun, setSelectedRun] = useState(null)
  const [statusFilter, setStatusFilter] = useState('all')
  const [loading, setLoading] = useState(true)
  const [lastRefresh, setLastRefresh] = useState(null)

  const fetchData = useCallback(async () => {
    try {
      const params = statusFilter !== 'all' ? `?status=${statusFilter}` : ''
      const [runsData, statsData] = await Promise.all([
        fetch(`${API}/runs${params}`).then(r => r.json()),
        fetch(`${API}/stats`).then(r => r.json()),
      ])
      setRuns(runsData)
      setStats(statsData)
      setLastRefresh(new Date().toLocaleTimeString('en-GB', { hour12: false }))
    } catch (e) {
      console.error('Fetch error:', e)
    } finally {
      setLoading(false)
    }
  }, [statusFilter])

  useEffect(() => {
    fetchData()
    const interval = setInterval(fetchData, 5000)
    return () => clearInterval(interval)
  }, [fetchData])

  return (
    <>
      <style>{css}</style>
      <div className="shell">
        <div className="topbar">
          <div className="topbar-brand">
            <div className="brand-dot" />
            LANGGRAPH MONITOR
          </div>
          <div className="topbar-right">
            {lastRefresh && <span>last refresh {lastRefresh}</span>}
            <button className="refresh-btn" onClick={fetchData}>↻ refresh</button>
          </div>
        </div>

        <StatBar stats={stats} />

        <div className="content">
          <div className="panel-left">
            <div className="panel-header">
              <span>Runs</span>
              <span>{runs.length}</span>
            </div>
            <div className="filter-row">
              {STATUS_LABELS.map(s => (
                <button
                  key={s}
                  className={`filter-btn ${statusFilter === s ? 'active' : ''}`}
                  onClick={() => { setStatusFilter(s); setSelectedRun(null) }}
                >
                  {s}
                </button>
              ))}
            </div>
            <div className="runs-list">
              {loading
                ? <div className="loading">loading...</div>
                : runs.length === 0
                  ? <div className="loading">no runs found</div>
                  : runs.map(r => (
                    <RunItem
                      key={r.id}
                      run={r}
                      selected={selectedRun === r.id}
                      onClick={() => setSelectedRun(r.id)}
                    />
                  ))
              }
            </div>
          </div>

          <div className="panel-right">
            <RunDetail runId={selectedRun} />
          </div>
        </div>
      </div>
    </>
  )
}
