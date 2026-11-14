
import { useEffect, useMemo, useState } from 'react'
import { dashboardApi } from '../services/api'

type Row = Record<string, unknown>
type Tab = 'optimized' | 'unscheduled' | 'windows' | 'weekly' | 'monthly'

const tabs: { id: Tab; label: string; title: string }[] = [
  { id: 'optimized', label: 'Optimized Schedule', title: 'Scheduled maintenance' },
  { id: 'unscheduled', label: 'Unscheduled', title: 'Requests requiring further planning' },
  { id: 'windows', label: 'Block Windows', title: 'Available block windows' },
  { id: 'weekly', label: 'Weekly Plan', title: 'Weekly maintenance plan' },
  { id: 'monthly', label: 'Monthly Plan', title: 'Monthly maintenance plan' },
]

function extractRows(value: unknown): Row[] {
  if (Array.isArray(value)) {
    return value.filter(item => item && typeof item === 'object') as Row[]
  }

  if (!value || typeof value !== 'object') return []

  const obj = value as Record<string, unknown>

  for (const key of ['items', 'records', 'results', 'data', 'schedule', 'requests', 'windows']) {
    if (Array.isArray(obj[key])) return extractRows(obj[key])
  }

  for (const nested of Object.values(obj)) {
    if (Array.isArray(nested)) {
      const rows = extractRows(nested)
      if (rows.length) return rows
    }
  }

  return []
}

function formatLabel(value: string) {
  return value
    .replace(/_/g, ' ')
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    .replace(/\b\w/g, char => char.toUpperCase())
}

function formatValue(value: unknown): string {
  if (value === null || value === undefined) return '—'
  if (typeof value === 'boolean') return value ? 'Yes' : 'No'
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}

export default function BlockPlanning() {
  const [activeTab, setActiveTab] = useState<Tab>('optimized')
  const [data, setData] = useState<Record<Tab, unknown>>({
    optimized: null,
    unscheduled: null,
    windows: null,
    weekly: null,
    monthly: null,
  })
  const [loading, setLoading] = useState(true)
  const [errors, setErrors] = useState<string[]>([])
  const [search, setSearch] = useState('')

  useEffect(() => {
    let mounted = true

    async function loadData() {
      setLoading(true)
      setErrors([])

      const requests: { key: Tab; request: Promise<any> }[] = [
        { key: 'optimized', request: dashboardApi.getOptimizedSchedule() },
        { key: 'unscheduled', request: dashboardApi.getUnscheduled() },
        { key: 'windows', request: dashboardApi.getBlockWindows() },
        { key: 'weekly', request: dashboardApi.getWeeklyPlan() },
        { key: 'monthly', request: dashboardApi.getMonthlyPlan() },
      ]

      const results = await Promise.allSettled(
        requests.map(item => item.request)
      )

      if (!mounted) return

      const nextData: Record<Tab, unknown> = {
        optimized: null,
        unscheduled: null,
        windows: null,
        weekly: null,
        monthly: null,
      }

      const failed: string[] = []

      results.forEach((result, index) => {
        const key = requests[index].key

        if (result.status === 'fulfilled') {
          nextData[key] = result.value.data
        } else {
          failed.push(tabs.find(tab => tab.id === key)?.label || key)
        }
      })

      setData(nextData)
      setErrors(failed)
      setLoading(false)
    }

    loadData()

    return () => {
      mounted = false
    }
  }, [])

  const currentRows = useMemo(
    () => extractRows(data[activeTab]),
    [data, activeTab]
  )

  const filteredRows = useMemo(() => {
    const query = search.trim().toLowerCase()
    if (!query) return currentRows

    return currentRows.filter(row =>
      Object.values(row).some(value =>
        formatValue(value).toLowerCase().includes(query)
      )
    )
  }, [currentRows, search])

  const currentTab = tabs.find(tab => tab.id === activeTab)!
  const columns = currentRows.length
    ? Object.keys(currentRows[0]).slice(0, 9)
    : []

  const counts = {
    optimized: extractRows(data.optimized).length,
    unscheduled: extractRows(data.unscheduled).length,
    windows: extractRows(data.windows).length,
  }

  return (
    <section className="page-section">
      <div className="page-heading">
        <div>
          <p className="section-eyebrow">OPERATIONS / SCHEDULING</p>
          <h1>Block Planning</h1>
          <p>
            Review optimized maintenance schedules, unscheduled requests
            and available block windows.
          </p>
        </div>
        <div className="data-source-label">SIMULATED PLANNING DATA</div>
      </div>

      <div className="stats-grid">
        <div className="stat-card">
          <span>Scheduled records</span>
          <strong>{loading ? '—' : counts.optimized}</strong>
        </div>
        <div className="stat-card">
          <span>Unscheduled records</span>
          <strong>{loading ? '—' : counts.unscheduled}</strong>
        </div>
        <div className="stat-card">
          <span>Available block windows</span>
          <strong>{loading ? '—' : counts.windows}</strong>
        </div>
      </div>

      <div className="panel">
        <div className="panel-heading">
          <div>
            <h2>Maintenance schedule register</h2>
            <p>Explore planning outputs generated by the optimizer.</p>
          </div>
        </div>

        <div className="planning-tabs">
          {tabs.map(tab => (
            <button
              key={tab.id}
              type="button"
              className={`planning-tab ${activeTab === tab.id ? 'active' : ''}`}
              onClick={() => {
                setActiveTab(tab.id)
                setSearch('')
              }}
            >
              {tab.label}
            </button>
          ))}
        </div>

        <div className="panel-heading planning-content-heading">
          <div>
            <h2>{currentTab.title}</h2>
            <p>
              {loading
                ? 'Loading planning data...'
                : `${filteredRows.length} records displayed`}
            </p>
          </div>

          <input
            className="planning-search"
            type="search"
            placeholder="Search records..."
            value={search}
            onChange={event => setSearch(event.target.value)}
          />
        </div>

        {loading ? (
          <div className="loading-state">Loading planning data...</div>
        ) : (
          <>
            {errors.includes(currentTab.label) && (
              <div className="error-state">
                Unable to load {currentTab.label}. Check the backend API.
              </div>
            )}

            {filteredRows.length > 0 ? (
              <div className="table-wrap">
                <table className="data-table">
                  <thead>
                    <tr>
                      {columns.map(column => (
                        <th key={column}>{formatLabel(column)}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {filteredRows.map((row, index) => (
                      <tr key={String(row.request_id || row.window_id || index)}>
                        {columns.map(column => (
                          <td key={column}>
                            {formatValue(row[column])}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="empty-state">
                {errors.includes(currentTab.label)
                  ? 'This planning endpoint could not be reached.'
                  : 'No tabular records were returned for this section.'}
              </div>
            )}
          </>
        )}
      </div>

      <div className="simulation-notice">
        Planning results are generated from simulated railway data.
        They are intended for prototype decision support and are not
        authorized operational railway block instructions.
      </div>
    </section>
  )
}
