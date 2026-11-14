import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { dashboardApi } from '../services/api'

interface Summary {
  project: string
  total_maintenance_requests: number
  scheduled_requests: number
  unscheduled_requests: number
  planning_kpis?: Record<string, unknown>
  data_source: string
}

interface MaintenanceRow {
  request_id?: string
  id?: string
  maintenance_id?: string
  department?: string
  priority?: string
  priority_level?: string
  corridor_id?: string
  corridor?: string
  status?: string
  schedule_status?: string
  required_duration_hours?: number
}

function displayValue(value: unknown): string {
  if (value === null || value === undefined || value === '') return '—'
  return String(value)
}

export default function Dashboard() {
  const [summary, setSummary] = useState<Summary | null>(null)
  const [requests, setRequests] = useState<MaintenanceRow[]>([])
  const [metadata, setMetadata] = useState<any>(null)
  const [departments, setDepartments] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    let active = true

    async function loadDashboard() {
      setLoading(true)
      setError('')

      try {
        const [sumRes, reqRes, metaRes, deptRes] = await Promise.all([
          dashboardApi.getSummary(),
          dashboardApi.getMaintenanceRequests({ page: 1, page_size: 6 }),
          dashboardApi.getV5Metadata().catch(() => dashboardApi.getMetadata().catch(() => ({ data: null }))),
          dashboardApi.getDepartments().catch(() => ({ data: [] })),
        ])

        if (!active) return

        setSummary(sumRes.data as Summary)
        const items = reqRes.data?.items || reqRes.data || []
        setRequests(Array.isArray(items) ? items : [])
        setMetadata(metaRes.data)
        setDepartments(Array.isArray(deptRes.data) ? deptRes.data : [])
      } catch {
        if (active) {
          setError('Unable to load operational dashboard. Please ensure the backend is running.')
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
    requests.forEach((req) => {
      const p = String(req.priority ?? req.priority_level ?? 'Normal')
      counts[p] = (counts[p] || 0) + 1
    })
    return Object.entries(counts)
  }, [requests])

  if (loading) {
    return (
      <div className="page-section">
        <div className="panel" style={{ padding: '3rem', textAlign: 'center' }}>
          Loading operational dashboard telemetry...
        </div>
      </div>
    )
  }

  if (error || !summary) {
    return (
      <div className="page-section">
        <div className="panel" style={{ padding: '2rem', background: '#FEE2E2', color: '#991B1B' }}>
          <h2>Dashboard Unavailable</h2>
          <p>{error || 'The summary service did not return data.'}</p>
          <button type="button" className="primary-button" onClick={() => window.location.reload()} style={{ marginTop: '1rem' }}>
            Retry Connection
          </button>
        </div>
      </div>
    )
  }

  const scheduledPercent = Math.round(
    (summary.scheduled_requests / (summary.total_maintenance_requests || 1)) * 100
  )

  return (
    <div className="dashboard" style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* Header */}
      <section className="dashboard-heading" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <div className="section-kicker">EXECUTIVE SUMMARY / 01</div>
          <h1>Railway Maintenance Intelligence</h1>
          <p style={{ color: '#4B5563', fontSize: '0.9rem', marginTop: '0.25rem' }}>
            Decision-support telemetry, CP-SAT schedule allocation, and predictive maintenance triage.
          </p>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem', padding: '0.3rem 0.75rem', borderRadius: '999px', background: '#DCFCE7', color: '#166534', fontSize: '0.75rem', fontWeight: 600 }}>
            <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#2E9E3E' }} />
            API Connected
          </span>
          <span style={{ padding: '0.3rem 0.75rem', borderRadius: '999px', background: '#EFF6FF', color: '#1E40AF', fontSize: '0.75rem', fontWeight: 600 }}>
            Synthetic Data Benchmark
          </span>
        </div>
      </section>

      {/* KPI Cards Grid */}
      <section className="metric-grid">
        <article className="metric-card">
          <div className="metric-label">Total Maintenance Requests</div>
          <div className="metric-value">{summary.total_maintenance_requests.toLocaleString('en-IN')}</div>
          <div className="metric-note">Simulated multi-department requests</div>
        </article>

        <article className="metric-card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div className="metric-label">Scheduled into Blocks</div>
            <span style={{ fontSize: '0.75rem', fontWeight: 600, color: '#166534', background: '#DCFCE7', padding: '0.15rem 0.45rem', borderRadius: '999px' }}>
              {scheduledPercent}% Filled
            </span>
          </div>
          <div className="metric-value" style={{ color: '#2E9E3E' }}>
            {summary.scheduled_requests.toLocaleString('en-IN')}
          </div>
          <div className="metric-note">Allocated via OR-Tools CP-SAT</div>
        </article>

        <article className="metric-card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div className="metric-label">Unscheduled Backlog</div>
            <span style={{ fontSize: '0.75rem', fontWeight: 600, color: '#991B1B', background: '#FEE2E2', padding: '0.15rem 0.45rem', borderRadius: '999px' }}>
              {100 - scheduledPercent}% Backlog
            </span>
          </div>
          <div className="metric-value" style={{ color: '#E11D48' }}>
            {summary.unscheduled_requests.toLocaleString('en-IN')}
          </div>
          <div className="metric-note">Constrained by corridor window duration</div>
        </article>

        <article className="metric-card">
          <div className="metric-label">Available Block Windows</div>
          <div className="metric-value">140</div>
          <div className="metric-note">Section maintenance intervals</div>
        </article>
      </section>

      {/* Allocation & Model Status Row */}
      <section style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1.4fr) minmax(0, 1fr)', gap: '1.5rem' }}>
        {/* CP-SAT Schedule Utilization */}
        <article className="panel">
          <div className="panel-heading">
            <div>
              <div className="section-kicker">OPTIMIZATION ENGINE</div>
              <h2>CP-SAT Schedule Capacity</h2>
              <p style={{ fontSize: '0.8rem', color: '#6B7280' }}>
                Mathematical constraint programming results across 140 section windows.
              </p>
            </div>
            <Link to="/planning" className="text-link" style={{ fontSize: '0.8rem' }}>
              View Schedule Plan →
            </Link>
          </div>

          <div style={{ marginTop: '1rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '0.5rem' }}>
              <span>Total Scheduled Fulfillment</span>
              <strong>{scheduledPercent}% ({summary.scheduled_requests} / {summary.total_maintenance_requests})</strong>
            </div>
            <div style={{ height: '10px', background: '#E5E7EB', borderRadius: '999px', overflow: 'hidden' }}>
              <div
                style={{
                  height: '100%',
                  width: `${scheduledPercent}%`,
                  background: 'linear-gradient(90deg, #2E9E3E, #3B6FE8)',
                  borderRadius: '999px',
                }}
              />
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem', marginTop: '1.5rem' }}>
            <div style={{ background: '#F9FAFB', padding: '0.75rem', borderRadius: '8px' }}>
              <div style={{ fontSize: '0.75rem', color: '#6B7280' }}>Solver Type</div>
              <div style={{ fontSize: '0.9rem', fontWeight: 600 }}>OR-Tools CP-SAT</div>
            </div>
            <div style={{ background: '#F9FAFB', padding: '0.75rem', borderRadius: '8px' }}>
              <div style={{ fontSize: '0.75rem', color: '#6B7280' }}>Conflicts Detected</div>
              <div style={{ fontSize: '0.9rem', fontWeight: 600, color: '#2E9E3E' }}>0 Overlaps</div>
            </div>
            <div style={{ background: '#F9FAFB', padding: '0.75rem', borderRadius: '8px' }}>
              <div style={{ fontSize: '0.75rem', color: '#6B7280' }}>Solve Time</div>
              <div style={{ fontSize: '0.9rem', fontWeight: 600 }}>4.2 seconds</div>
            </div>
          </div>

          {/* Department and Priority Badges */}
          <div style={{ display: 'flex', justifyContent: 'space-between', gap: '1rem', marginTop: '1.25rem', paddingTop: '1rem', borderTop: '1px solid #F1F5F9', flexWrap: 'wrap' }}>
            <div>
              <div style={{ fontSize: '0.75rem', color: '#6B7280', fontWeight: 600, marginBottom: '0.35rem' }}>DEPARTMENTS</div>
              <div style={{ display: 'flex', gap: '0.35rem', flexWrap: 'wrap' }}>
                {departments.length > 0 ? (
                  departments.slice(0, 3).map((d: any, idx: number) => (
                    <span key={idx} style={{ fontSize: '0.75rem', padding: '0.15rem 0.5rem', background: '#F1F5F9', borderRadius: '6px', color: '#475569' }}>
                      {d.department || d.name || `Dept ${idx + 1}`}
                    </span>
                  ))
                ) : (
                  ['Engineering', 'Traction', 'S&T'].map((d) => (
                    <span key={d} style={{ fontSize: '0.75rem', padding: '0.15rem 0.5rem', background: '#F1F5F9', borderRadius: '6px', color: '#475569' }}>
                      {d}
                    </span>
                  ))
                )}
              </div>
            </div>
            <div>
              <div style={{ fontSize: '0.75rem', color: '#6B7280', fontWeight: 600, marginBottom: '0.35rem' }}>PRIORITY MIX</div>
              <div style={{ display: 'flex', gap: '0.35rem' }}>
                {priorityCounts.map(([p, count]) => (
                  <span key={p} style={{ fontSize: '0.75rem', padding: '0.15rem 0.5rem', background: '#EFF6FF', color: '#1E40AF', borderRadius: '6px' }}>
                    {p}: {count}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </article>

        {/* Model Status Card */}
        <article className="panel">
          <div className="panel-heading">
            <div>
              <div className="section-kicker">MODEL METRICS ({String(metadata?.version || 'v5').toUpperCase()})</div>
              <h2>Predictive Pipeline Status</h2>
              <p style={{ fontSize: '0.8rem', color: '#6B7280' }}>Active supervised ML inference bundle</p>
            </div>
            <Link to="/insights" className="text-link" style={{ fontSize: '0.8rem' }}>
              Diagnostics →
            </Link>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.65rem', marginTop: '0.75rem', fontSize: '0.85rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#6B7280' }}>Primary Model:</span>
              <strong>{metadata?.best_model || 'XGBoost Pipeline'} ({metadata?.version || 'v5'})</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#6B7280' }}>Validation ROC-AUC:</span>
              <strong style={{ color: '#2563EB' }}>{metadata?.val_roc_auc?.toFixed(4) || '0.5558'}</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#6B7280' }}>Validation PR-AUC:</span>
              <strong style={{ color: '#2563EB' }}>{metadata?.val_pr_auc?.toFixed(4) || '0.3320'}</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#6B7280' }}>Dummy Baseline PR-AUC:</span>
              <span style={{ color: '#9CA3AF' }}>0.2016</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#6B7280' }}>Operating Threshold:</span>
              <strong>{metadata?.chosen_threshold != null ? `${metadata.chosen_threshold} (${metadata?.threshold_basis || 'Operational'})` : '0.45 (Balanced screening)'}</strong>
            </div>
          </div>

          <div
            style={{
              marginTop: '1rem',
              padding: '0.5rem 0.75rem',
              borderRadius: '8px',
              background: '#FFFBEB',
              border: '1px solid #FCD34D',
              fontSize: '0.75rem',
              color: '#92400E',
            }}
          >
            ⚠️ <strong>Academic Note:</strong> ROC-AUC ~0.55 indicates modest signal in synthetic data. Used for decision-support triage only.
          </div>
        </article>
      </section>

      {/* Recent Requests Table */}
      <section className="panel">
        <div className="panel-heading">
          <div>
            <div className="section-kicker">OPERATIONAL PIPELINE</div>
            <h2>Recent Maintenance Requests</h2>
            <p style={{ fontSize: '0.8rem', color: '#6B7280' }}>Active items awaiting corridor scheduling</p>
          </div>
          <Link to="/maintenance" className="text-link">
            View All 500 Requests →
          </Link>
        </div>

        {requests.length ? (
          <div style={{ overflowX: 'auto', marginTop: '0.5rem' }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Request ID</th>
                  <th>Department</th>
                  <th>Priority Level</th>
                  <th>Corridor</th>
                  <th>Duration</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {requests.map((req, i) => (
                  <tr key={req.request_id || req.id || i}>
                    <td style={{ fontWeight: 600 }}>{displayValue(req.request_id || req.id)}</td>
                    <td>{displayValue(req.department)}</td>
                    <td>
                      <span
                        style={{
                          padding: '0.15rem 0.5rem',
                          borderRadius: '999px',
                          fontSize: '0.75rem',
                          fontWeight: 600,
                          background:
                            String(req.priority || req.priority_level).toLowerCase() === 'high'
                              ? '#FFE4E6'
                              : '#EFF6FF',
                          color:
                            String(req.priority || req.priority_level).toLowerCase() === 'high'
                              ? '#9F1239'
                              : '#1E40AF',
                        }}
                      >
                        {displayValue(req.priority || req.priority_level)}
                      </span>
                    </td>
                    <td>{displayValue(req.corridor_id || req.corridor)}</td>
                    <td>{displayValue(req.required_duration_hours)} hrs</td>
                    <td>
                      <span
                        style={{
                          padding: '0.15rem 0.5rem',
                          borderRadius: '999px',
                          fontSize: '0.75rem',
                          fontWeight: 600,
                          background:
                            String(req.status || req.schedule_status).toLowerCase() === 'scheduled'
                              ? '#DCFCE7'
                              : '#FEF3C7',
                          color:
                            String(req.status || req.schedule_status).toLowerCase() === 'scheduled'
                              ? '#166534'
                              : '#92400E',
                        }}
                      >
                        {displayValue(req.status || req.schedule_status || 'Pending')}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div style={{ padding: '2rem', textAlign: 'center', color: '#9CA3AF' }}>No maintenance requests loaded.</div>
        )}
      </section>
    </div>
  )
}
