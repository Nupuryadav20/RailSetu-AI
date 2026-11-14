import { useEffect, useMemo, useState } from 'react'
import { dashboardApi } from '../services/api'

type Summary = {
  project: string
  total_maintenance_requests: number
  scheduled_requests: number
  unscheduled_requests: number
  planning_kpis?: Record<string, unknown>
  data_source: string
}

type MaintenanceRow = Record<string, unknown>

function getRows(response: unknown): MaintenanceRow[] {
  if (Array.isArray(response)) return response as MaintenanceRow[]

  if (response && typeof response === 'object') {
    const data = response as Record<string, unknown>
    for (const key of ['records', 'items', 'results', 'data', 'requests']) {
      if (Array.isArray(data[key])) return data[key] as MaintenanceRow[]
    }
  }
  return []
}

function displayValue(value: unknown): string {
  if (value === null || value === undefined || value === '') return '—'
  return String(value)
}

export default function Dashboard() {
  const [summary, setSummary] = useState<Summary | null>(null)
  const [requests, setRequests] = useState<MaintenanceRow[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    let active = true

    async function loadDashboard() {
      setLoading(true)
      setError('')

      try {
        const [summaryResponse, requestsResponse] = await Promise.all([
          dashboardApi.getSummary(),
          dashboardApi.getMaintenanceRequests({ page: 1, page_size: 8 }),
        ])

        if (!active) return

        setSummary(summaryResponse.data as Summary)
        setRequests(getRows(requestsResponse.data))
      } catch {
        if (active) {
          setError(
            'Unable to load dashboard data. Check that the RailSetu AI backend is running.',
          )
        }
      } finally {
        if (active) setLoading(false)
      }
    }

    loadDashboard()
    return () => {
      active = false
    }
  }, [])

  const priorityCounts = useMemo(() => {
    const counts: Record<string, number> = {}
    requests.forEach((request) => {
      const priority = String(
        request.priority ?? request.priority_level ?? 'Unspecified',
      )
      counts[priority] = (counts[priority] || 0) + 1
    })
    return Object.entries(counts)
  }, [requests])

  if (loading) {
    return <div className="dashboard-state">Loading operational data...</div>
  }

  if (error || !summary) {
    return (
      <div className="dashboard-state dashboard-error">
        <h2>Dashboard unavailable</h2>
        <p>{error || 'The dashboard summary was not returned by the server.'}</p>
        <button onClick={() => window.location.reload()}>Retry connection</button>
      </div>
    )
  }

  const metrics = [
    {
      label: 'Total maintenance requests',
      value: summary.total_maintenance_requests,
      note: 'Generated maintenance requests',
    },
    {
      label: 'Scheduled requests',
      value: summary.scheduled_requests,
      note: 'Included in optimized schedule',
    },
    {
      label: 'Unscheduled requests',
      value: summary.unscheduled_requests,
      note: 'Require further planning',
    },
  ]

  return (
    <div className="dashboard">
      <section className="dashboard-heading">
        <div>
          <div className="section-kicker">SYSTEM OVERVIEW / 01</div>
          <h1>Operations overview</h1>
          <p>
            Maintenance activity, scheduling status and operational planning.
          </p>
        </div>
        <div className="live-label">
          <span className="status-dot" />
          Backend connected
        </div>
      </section>

      <section className="metric-grid">
        {metrics.map((metric, index) => (
          <article className="metric-card" key={metric.label}>
            <div className="metric-index">0{index + 1}</div>
            <div className="metric-label">{metric.label}</div>
            <div className="metric-value">
              {metric.value.toLocaleString('en-IN')}
            </div>
            <div className="metric-note">{metric.note}</div>
          </article>
        ))}
      </section>

      <section className="dashboard-grid">
        <article className="panel planning-panel">
          <div className="panel-heading">
            <div>
              <div className="section-kicker">PLANNING</div>
              <h2>Planning summary</h2>
            </div>
            <span className="panel-tag">Current generated plan</span>
          </div>

          {summary.planning_kpis &&
          Object.keys(summary.planning_kpis).length > 0 ? (
            <div className="planning-values">
              {Object.entries(summary.planning_kpis).map(([key, value]) => (
                <div className="planning-row" key={key}>
                  <span>{key.replaceAll('_', ' ')}</span>
                  <strong>{displayValue(value)}</strong>
                </div>
              ))}
            </div>
          ) : (
            <p className="muted-message">No planning KPI details available.</p>
          )}
        </article>

        <article className="panel priority-panel">
          <div className="panel-heading">
            <div>
              <div className="section-kicker">REQUEST ANALYSIS</div>
              <h2>Priority distribution</h2>
            </div>
            <span className="panel-tag">Loaded records</span>
          </div>

          {priorityCounts.length ? (
            <div className="priority-list">
              {priorityCounts.map(([priority, count]) => (
                <div className="priority-row" key={priority}>
                  <div className="priority-row-label">
                    <span>{priority}</span>
                    <strong>{count}</strong>
                  </div>
                  <div className="priority-track">
                    <div
                      className="priority-fill"
                      style={{
                        width: `${(count / requests.length) * 100}%`,
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="muted-message">
              No maintenance request records available.
            </p>
          )}
        </article>
      </section>

      <section className="panel requests-panel">
        <div className="panel-heading">
          <div>
            <div className="section-kicker">MAINTENANCE REGISTER</div>
            <h2>Recent maintenance requests</h2>
          </div>
          <a className="text-link" href="/maintenance">
            View all requests →
          </a>
        </div>

        {requests.length ? (
          <div className="table-scroll">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Request ID</th>
                  <th>Department</th>
                  <th>Priority</th>
                  <th>Corridor</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {requests.map((request, index) => (
                  <tr key={String(request.request_id ?? request.id ?? index)}>
                    <td>
                      {displayValue(
                        request.request_id ?? request.id ?? request.maintenance_id,
                      )}
                    </td>
                    <td>{displayValue(request.department)}</td>
                    <td>
                      <span className="table-priority">
                        {displayValue(
                          request.priority ?? request.priority_level,
                        )}
                      </span>
                    </td>
                    <td>{displayValue(request.corridor)}</td>
                    <td>
                      {displayValue(request.status ?? request.schedule_status)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="muted-message">
            No request records were returned by the API.
          </p>
        )}
      </section>

      <footer className="dashboard-disclaimer">
        <span>DATA SOURCE</span>
        <p>{summary.data_source}</p>
        <p>
          Prototype decision-support system. Outputs are not official railway
          safety ratings or operational authorizations.
        </p>
      </footer>
    </div>
  )
}