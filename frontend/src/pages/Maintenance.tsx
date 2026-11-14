import { useCallback, useEffect, useState, useMemo } from 'react'
import { dashboardApi } from '../services/api'

interface MaintenanceRequest {
  request_id: string
  record_id: number
  asset_id: string
  corridor_id: string
  location_km: number
  department: string
  maintenance_type: string
  required_duration_hours: number
  team_size_required: number
  earliest_start: string
  deadline: string
  model_risk_probability_percent: number
  equipment_condition_score: number
  operational_priority_score: number
  priority_level: string
  status: string
  is_simulated: boolean
}

interface ApiResult {
  items: MaintenanceRequest[]
  pagination: {
    page: number
    page_size: number
    total_records: number
    total_pages: number
    has_next: boolean
    has_previous: boolean
  }
}

const DEPARTMENTS = ['Engineering', 'Traction', 'Signal & Telecom']
const PRIORITIES = ['High', 'Medium', 'Low']
const CORRIDORS = ['CORRIDOR-A', 'CORRIDOR-B', 'CORRIDOR-C', 'CORRIDOR-D', 'CORRIDOR-E']

const formatDate = (value: string) => {
  if (!value) return '—'
  const date = new Date(value.replace(' ', 'T'))
  return Number.isNaN(date.getTime())
    ? value
    : date.toLocaleDateString('en-IN', {
        day: '2-digit',
        month: 'short',
        year: 'numeric',
      })
}

const formatNumber = (value: number) =>
  Number.isFinite(value) ? value.toFixed(1) : '—'

export default function Maintenance() {
  const [items, setItems] = useState<MaintenanceRequest[]>([])
  const [pagination, setPagination] = useState<ApiResult['pagination'] | null>(null)

  const [search, setSearch] = useState('')
  const [department, setDepartment] = useState('')
  const [priority, setPriority] = useState('')
  const [corridor, setCorridor] = useState('')

  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(10)
  const [selected, setSelected] = useState<MaintenanceRequest | null>(null)

  const [sortField, setSortField] = useState<keyof MaintenanceRequest>('operational_priority_score')
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc')

  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  const fetchRequests = useCallback(async () => {
    setLoading(true)
    setError('')

    try {
      const response = await dashboardApi.getMaintenanceRequests({
        page,
        page_size: pageSize,
        search: search.trim() || undefined,
        department: department.trim() || undefined,
        priority: priority.trim() || undefined,
        corridor: corridor.trim() || undefined,
      })

      const result = response.data as ApiResult
      setItems(Array.isArray(result.items) ? result.items : [])
      setPagination(result.pagination ?? null)
    } catch (err) {
      console.error('Maintenance request loading failed:', err)
      setError('Unable to load maintenance requests. Check the backend connection and try again.')
      setItems([])
      setPagination(null)
    } finally {
      setLoading(false)
    }
  }, [page, pageSize, search, department, priority, corridor])

  useEffect(() => {
    const timer = window.setTimeout(() => {
      void fetchRequests()
    }, 250)
    return () => window.clearTimeout(timer)
  }, [fetchRequests])

  const resetFilters = () => {
    setSearch('')
    setDepartment('')
    setPriority('')
    setCorridor('')
    setPage(1)
    setSelected(null)
  }

  const handleSort = (field: keyof MaintenanceRequest) => {
    if (sortField === field) {
      setSortDir((prev) => (prev === 'asc' ? 'desc' : 'asc'))
    } else {
      setSortField(field)
      setSortDir('desc')
    }
  }

  const sortedItems = useMemo(() => {
    const copy = [...items]
    copy.sort((a, b) => {
      const valA = a[sortField]
      const valB = b[sortField]
      if (typeof valA === 'number' && typeof valB === 'number') {
        return sortDir === 'asc' ? valA - valB : valB - valA
      }
      return sortDir === 'asc'
        ? String(valA).localeCompare(String(valB))
        : String(valB).localeCompare(String(valA))
    })
    return copy
  }, [items, sortField, sortDir])

  const getPriorityBadgeClass = (lvl: string) => {
    switch (lvl?.toLowerCase()) {
      case 'high':
        return 'badge badge-priority-high'
      case 'medium':
        return 'badge badge-priority-medium'
      default:
        return 'badge badge-priority-low'
    }
  }

  const getStatusBadgeClass = (status: string) => {
    switch (status?.toLowerCase()) {
      case 'scheduled':
        return 'badge badge-status-scheduled'
      default:
        return 'badge badge-status-pending'
    }
  }

  const getRiskColor = (percent: number) => {
    if (percent >= 70) return '#E11D48'
    if (percent >= 40) return '#D97706'
    return '#2E9E3E'
  }

  return (
    <section className="page-section">
      <div className="page-heading">
        <div>
          <div className="eyebrow">OPERATIONS / REQUEST REGISTER</div>
          <h1>Maintenance Requests</h1>
          <p>Monitor, search, filter and inspect corridor maintenance requirements.</p>
        </div>
        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          <span style={{ fontSize: '0.75rem', fontWeight: 600, padding: '0.25rem 0.6rem', borderRadius: '999px', background: '#DCFCE7', color: '#166534' }}>
            SIMULATED PLANNING DATA
          </span>
        </div>
      </div>

      {/* 3 Header KPI Cards */}
      <div className="metric-grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))' }}>
        <div className="metric-card">
          <div className="metric-label">TOTAL MATCHING REQUESTS</div>
          <div className="metric-value">{pagination?.total_records ?? '—'}</div>
          <div className="metric-note">Backlog items meeting active filters</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">RECORDS ON THIS PAGE</div>
          <div className="metric-value">{items.length}</div>
          <div className="metric-note">Currently rendered in table view</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">CURRENT PAGE</div>
          <div className="metric-value">
            {pagination ? `${pagination.page} / ${pagination.total_pages || 1}` : '—'}
          </div>
          <div className="metric-note">Page navigation status</div>
        </div>
      </div>

      {/* Filter Card */}
      <div className="filter-card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
          <div>
            <h2 style={{ fontSize: '1rem', fontWeight: 600, color: '#0F172A', margin: 0 }}>Filter Requests</h2>
            <p style={{ fontSize: '0.8rem', color: '#64748B', margin: '2px 0 0 0' }}>Server-side query filtering</p>
          </div>
          <button type="button" className="secondary-button" onClick={resetFilters}>
            Clear Filters
          </button>
        </div>

        <div className="filter-grid">
          <div className="filter-group">
            <label className="filter-label">Search</label>
            <input
              type="text"
              className="filter-input"
              value={search}
              onChange={(e) => {
                setSearch(e.target.value)
                setPage(1)
              }}
              placeholder="Request ID, asset, or task..."
            />
          </div>

          <div className="filter-group">
            <label className="filter-label">Department</label>
            <select
              className="filter-select"
              value={department}
              onChange={(e) => {
                setDepartment(e.target.value)
                setPage(1)
              }}
            >
              <option value="">All Departments</option>
              {DEPARTMENTS.map((dept) => (
                <option key={dept} value={dept}>
                  {dept}
                </option>
              ))}
            </select>
          </div>

          <div className="filter-group">
            <label className="filter-label">Priority</label>
            <select
              className="filter-select"
              value={priority}
              onChange={(e) => {
                setPriority(e.target.value)
                setPage(1)
              }}
            >
              <option value="">All Priorities</option>
              {PRIORITIES.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
          </div>

          <div className="filter-group">
            <label className="filter-label">Corridor</label>
            <select
              className="filter-select"
              value={corridor}
              onChange={(e) => {
                setCorridor(e.target.value)
                setPage(1)
              }}
            >
              <option value="">All Corridors</option>
              {CORRIDORS.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Main Table Panel */}
      <div className="panel" style={{ padding: 0, overflow: 'hidden' }}>
        {error && (
          <div style={{ padding: '16px', background: '#FEE2E2', color: '#991B1B', fontSize: '0.85rem' }}>
            {error}
          </div>
        )}

        <div className="table-wrap" style={{ maxHeight: '600px' }}>
          <table className="data-table">
            <thead>
              <tr>
                <th onClick={() => handleSort('request_id')} style={{ cursor: 'pointer' }}>
                  REQUEST ID {sortField === 'request_id' && (sortDir === 'asc' ? '↑' : '↓')}
                </th>
                <th onClick={() => handleSort('maintenance_type')} style={{ cursor: 'pointer' }}>
                  MAINTENANCE ACTIVITY {sortField === 'maintenance_type' && (sortDir === 'asc' ? '↑' : '↓')}
                </th>
                <th onClick={() => handleSort('department')} style={{ cursor: 'pointer' }}>
                  DEPARTMENT {sortField === 'department' && (sortDir === 'asc' ? '↑' : '↓')}
                </th>
                <th onClick={() => handleSort('corridor_id')} style={{ cursor: 'pointer' }}>
                  CORRIDOR {sortField === 'corridor_id' && (sortDir === 'asc' ? '↑' : '↓')}
                </th>
                <th onClick={() => handleSort('model_risk_probability_percent')} style={{ cursor: 'pointer' }}>
                  ML RISK {sortField === 'model_risk_probability_percent' && (sortDir === 'asc' ? '↑' : '↓')}
                </th>
                <th onClick={() => handleSort('operational_priority_score')} style={{ cursor: 'pointer' }}>
                  PRIORITY {sortField === 'operational_priority_score' && (sortDir === 'asc' ? '↑' : '↓')}
                </th>
                <th onClick={() => handleSort('status')} style={{ cursor: 'pointer' }}>
                  STATUS {sortField === 'status' && (sortDir === 'asc' ? '↑' : '↓')}
                </th>
                <th onClick={() => handleSort('deadline')} style={{ cursor: 'pointer' }}>
                  DEADLINE {sortField === 'deadline' && (sortDir === 'asc' ? '↑' : '↓')}
                </th>
                <th style={{ textAlign: 'right' }}>ACTION</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr>
                  <td colSpan={9} style={{ textAlign: 'center', padding: '3rem', color: '#64748B' }}>
                    Loading maintenance records...
                  </td>
                </tr>
              ) : sortedItems.length === 0 ? (
                <tr>
                  <td colSpan={9} style={{ textAlign: 'center', padding: '3rem', color: '#64748B' }}>
                    No maintenance requests match the current filters.
                  </td>
                </tr>
              ) : (
                sortedItems.map((item) => (
                  <tr key={item.request_id}>
                    <td>
                      <div style={{ fontWeight: 600, color: '#0F172A' }}>{item.request_id}</div>
                      <div style={{ fontSize: '0.75rem', color: '#64748B' }}>{item.asset_id}</div>
                    </td>
                    <td>
                      <div style={{ fontWeight: 500, color: '#1E293B' }}>{item.maintenance_type}</div>
                      <div style={{ fontSize: '0.75rem', color: '#64748B' }}>{item.required_duration_hours}h • {item.team_size_required} crew</div>
                    </td>
                    <td>{item.department}</td>
                    <td>
                      <span style={{ fontSize: '0.8rem', padding: '2px 6px', background: '#F1F5F9', borderRadius: '4px' }}>
                        {item.corridor_id}
                      </span>
                    </td>
                    <td>
                      <div className="risk-bar-container">
                        <div className="risk-bar-track">
                          <div
                            className="risk-bar-fill"
                            style={{
                              width: `${Math.min(100, Math.max(5, item.model_risk_probability_percent))}%`,
                              backgroundColor: getRiskColor(item.model_risk_probability_percent),
                            }}
                          />
                        </div>
                        <span style={{ fontSize: '0.8rem', fontWeight: 600, minWidth: '40px' }}>
                          {formatNumber(item.model_risk_probability_percent)}%
                        </span>
                      </div>
                    </td>
                    <td>
                      <span className={getPriorityBadgeClass(item.priority_level)}>
                        {item.priority_level} ({formatNumber(item.operational_priority_score)})
                      </span>
                    </td>
                    <td>
                      <span className={getStatusBadgeClass(item.status)}>
                        {item.status}
                      </span>
                    </td>
                    <td>{formatDate(item.deadline)}</td>
                    <td style={{ textAlign: 'right' }}>
                      <button
                        type="button"
                        className="secondary-button"
                        style={{ padding: '4px 10px', fontSize: '0.75rem' }}
                        onClick={() => setSelected(item)}
                      >
                        Details
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination Bar */}
        <div className="pagination-bar">
          <div>
            Showing <strong>{items.length}</strong> of <strong>{pagination?.total_records ?? 0}</strong> requests
          </div>

          <div className="pagination-controls">
            <span style={{ marginRight: '8px' }}>Rows per page:</span>
            <select
              value={pageSize}
              onChange={(e) => {
                setPageSize(Number(e.target.value))
                setPage(1)
              }}
              style={{
                padding: '4px 8px',
                borderRadius: '6px',
                border: '1px solid #CBD5E1',
                fontSize: '0.8rem',
                marginRight: '16px',
              }}
            >
              <option value={10}>10</option>
              <option value={20}>20</option>
              <option value={50}>50</option>
            </select>

            <button
              type="button"
              className="pagination-btn"
              disabled={!pagination?.has_previous}
              onClick={() => setPage((p) => Math.max(1, p - 1))}
            >
              Previous
            </button>
            <span style={{ padding: '0 8px', fontWeight: 600 }}>
              {pagination?.page ?? 1} / {pagination?.total_pages || 1}
            </span>
            <button
              type="button"
              className="pagination-btn"
              disabled={!pagination?.has_next}
              onClick={() => setPage((p) => p + 1)}
            >
              Next
            </button>
          </div>
        </div>
      </div>

      {/* Row Details Drawer */}
      {selected && (
        <>
          <div className="drawer-backdrop" onClick={() => setSelected(null)} />
          <div className="drawer-panel">
            <div className="drawer-header">
              <div>
                <div className="eyebrow">{selected.corridor_id} • KM {selected.location_km}</div>
                <h3>{selected.request_id}</h3>
                <div style={{ fontSize: '0.8rem', color: '#64748B', marginTop: '2px' }}>{selected.asset_id}</div>
              </div>
              <button
                type="button"
                className="secondary-button"
                style={{ padding: '4px 8px', fontSize: '0.8rem' }}
                onClick={() => setSelected(null)}
              >
                ✕ Close
              </button>
            </div>

            <div className="drawer-body">
              <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '8px' }}>
                <span className={getPriorityBadgeClass(selected.priority_level)}>
                  Priority: {selected.priority_level}
                </span>
                <span className={getStatusBadgeClass(selected.status)}>
                  Status: {selected.status}
                </span>
              </div>

              <div className="drawer-row">
                <span className="drawer-row-label">Maintenance Activity</span>
                <span className="drawer-row-val">{selected.maintenance_type}</span>
              </div>

              <div className="drawer-row">
                <span className="drawer-row-label">Department</span>
                <span className="drawer-row-val">{selected.department}</span>
              </div>

              <div className="drawer-row">
                <span className="drawer-row-label">Required Work Duration</span>
                <span className="drawer-row-val">{selected.required_duration_hours} hours</span>
              </div>

              <div className="drawer-row">
                <span className="drawer-row-label">Team Size Required</span>
                <span className="drawer-row-val">{selected.team_size_required} specialists</span>
              </div>

              <div className="drawer-row">
                <span className="drawer-row-label">Earliest Feasible Start</span>
                <span className="drawer-row-val">{formatDate(selected.earliest_start)}</span>
              </div>

              <div className="drawer-row">
                <span className="drawer-row-label">Operational Deadline</span>
                <span className="drawer-row-val">{formatDate(selected.deadline)}</span>
              </div>

              <div style={{ marginTop: '16px', padding: '16px', background: '#F8FAFC', borderRadius: '12px', border: '1px solid #E2E8F0' }}>
                <div style={{ fontSize: '0.75rem', fontWeight: 600, color: '#334155', textTransform: 'uppercase', marginBottom: '8px' }}>
                  Risk & Priority Architecture
                </div>
                <div className="drawer-row">
                  <span className="drawer-row-label">Concept A: ML Failure Prob</span>
                  <span className="drawer-row-val" style={{ color: getRiskColor(selected.model_risk_probability_percent) }}>
                    {formatNumber(selected.model_risk_probability_percent)}%
                  </span>
                </div>
                <div className="drawer-row">
                  <span className="drawer-row-label">Equipment Condition Score</span>
                  <span className="drawer-row-val">{formatNumber(selected.equipment_condition_score)} / 100</span>
                </div>
                <div className="drawer-row" style={{ borderBottom: 'none' }}>
                  <span className="drawer-row-label">Concept C: Operational Priority</span>
                  <span className="drawer-row-val" style={{ color: '#2563EB', fontSize: '1rem' }}>
                    {formatNumber(selected.operational_priority_score)} / 100
                  </span>
                </div>
              </div>
            </div>
          </div>
        </>
      )}
    </section>
  )
}
