
import { useCallback, useEffect, useState } from 'react';
import { dashboardApi } from '../services/api';

interface MaintenanceRequest {
  request_id: string;
  record_id: number;
  asset_id: string;
  corridor_id: string;
  location_km: number;
  department: string;
  maintenance_type: string;
  required_duration_hours: number;
  team_size_required: number;
  earliest_start: string;
  deadline: string;
  model_risk_probability_percent: number;
  equipment_condition_score: number;
  operational_priority_score: number;
  priority_level: string;
  status: string;
  is_simulated: boolean;
}

interface ApiResult {
  items: MaintenanceRequest[];
  pagination: {
    page: number;
    page_size: number;
    total_records: number;
    total_pages: number;
    has_next: boolean;
    has_previous: boolean;
  };
}

const formatDate = (value: string) => {
  if (!value) return '—';
  const date = new Date(value.replace(' ', 'T'));
  return Number.isNaN(date.getTime())
    ? value
    : date.toLocaleDateString('en-IN', {
        day: '2-digit',
        month: 'short',
        year: 'numeric',
      });
};

const formatNumber = (value: number) =>
  Number.isFinite(value) ? value.toFixed(1) : '—';

export default function Maintenance() {
  const [items, setItems] = useState<MaintenanceRequest[]>([]);
  const [pagination, setPagination] = useState<ApiResult['pagination'] | null>(null);

  const [search, setSearch] = useState('');
  const [department, setDepartment] = useState('');
  const [priority, setPriority] = useState('');
  const [corridor, setCorridor] = useState('');

  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const [selected, setSelected] = useState<MaintenanceRequest | null>(null);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const fetchRequests = useCallback(async () => {
    setLoading(true);
    setError('');

    try {
      const response = await dashboardApi.getMaintenanceRequests({
        page,
        page_size: pageSize,
        search: search.trim() || undefined,
        department: department.trim() || undefined,
        priority: priority.trim() || undefined,
        corridor: corridor.trim() || undefined,
      });

      const result = response.data as ApiResult;

      setItems(Array.isArray(result.items) ? result.items : []);
      setPagination(result.pagination ?? null);
    } catch (err) {
      console.error('Maintenance request loading failed:', err);
      setError('Unable to load maintenance requests. Check the backend connection and try again.');
      setItems([]);
      setPagination(null);
    } finally {
      setLoading(false);
    }
  }, [page, pageSize, search, department, priority, corridor]);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      void fetchRequests();
    }, 300);

    return () => window.clearTimeout(timer);
  }, [fetchRequests]);

  const resetFilters = () => {
    setSearch('');
    setDepartment('');
    setPriority('');
    setCorridor('');
    setPage(1);
    setSelected(null);
  };

  const updateFilter = (
    setter: (value: string) => void,
    value: string
  ) => {
    setter(value);
    setPage(1);
    setSelected(null);
  };

  const priorityClass = (value: string) =>
    `priority-label priority-${value.toLowerCase().replace(/\s+/g, '-')}`;

  const statusClass = (value: string) =>
    `status-label status-${value.toLowerCase().replace(/\s+/g, '-')}`;

  return (
    <section className="page-section maintenance-page">
      <div className="maintenance-heading">
        <div>
          <div className="section-eyebrow">OPERATIONS / REQUEST REGISTER</div>
          <h1>Maintenance Requests</h1>
          <p>Monitor, search and review maintenance activities across railway corridors.</p>
        </div>
        <div className="data-source-label">SIMULATED PLANNING DATA</div>
      </div>

      <div className="maintenance-summary">
        <div className="maintenance-stat">
          <span>TOTAL MATCHING REQUESTS</span>
          <strong>{pagination?.total_records ?? '—'}</strong>
        </div>
        <div className="maintenance-stat">
          <span>RECORDS ON THIS PAGE</span>
          <strong>{items.length}</strong>
        </div>
        <div className="maintenance-stat">
          <span>CURRENT PAGE</span>
          <strong>
            {pagination ? `${pagination.page} / ${pagination.total_pages}` : '—'}
          </strong>
        </div>
      </div>

      <div className="maintenance-panel">
        <div className="maintenance-panel-heading">
          <div>
            <h2>Request register</h2>
            <p>Filters are applied directly by the backend.</p>
          </div>
          <button className="secondary-button" onClick={resetFilters}>
            Clear filters
          </button>
        </div>

        <div className="maintenance-filters">
          <label className="filter-field search-field">
            <span>SEARCH</span>
            <input
              value={search}
              onChange={(e) => updateFilter(setSearch, e.target.value)}
              placeholder="Request ID, asset or maintenance type"
            />
          </label>

          <label className="filter-field">
            <span>DEPARTMENT</span>
            <input
              value={department}
              onChange={(e) => updateFilter(setDepartment, e.target.value)}
              placeholder="e.g. Engineering"
            />
          </label>

          <label className="filter-field">
            <span>PRIORITY</span>
            <input
              value={priority}
              onChange={(e) => updateFilter(setPriority, e.target.value)}
              placeholder="e.g. Medium"
            />
          </label>

          <label className="filter-field">
            <span>CORRIDOR</span>
            <input
              value={corridor}
              onChange={(e) => updateFilter(setCorridor, e.target.value)}
              placeholder="e.g. CORRIDOR-E"
            />
          </label>
        </div>

        {error && (
          <div className="maintenance-error">
            <strong>Request loading failed</strong>
            <p>{error}</p>
            <button className="secondary-button" onClick={() => void fetchRequests()}>
              Retry
            </button>
          </div>
        )}

        <div className="maintenance-table-wrap">
          <table className="maintenance-table">
            <thead>
              <tr>
                <th>REQUEST</th>
                <th>MAINTENANCE ACTIVITY</th>
                <th>DEPARTMENT</th>
                <th>CORRIDOR</th>
                <th>RISK</th>
                <th>PRIORITY</th>
                <th>STATUS</th>
                <th>DEADLINE</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr>
                  <td colSpan={9} className="table-message">
                    Loading maintenance records...
                  </td>
                </tr>
              ) : error ? (
                <tr>
                  <td colSpan={9} className="table-message">
                    Data unavailable.
                  </td>
                </tr>
              ) : items.length === 0 ? (
                <tr>
                  <td colSpan={9} className="table-message">
                    No maintenance requests match these filters.
                  </td>
                </tr>
              ) : (
                items.map((item) => (
                  <tr key={item.request_id}>
                    <td>
                      <strong className="request-id">{item.request_id}</strong>
                      <small>{item.asset_id}</small>
                    </td>
                    <td>
                      <strong>{item.maintenance_type}</strong>
                      <small>{item.required_duration_hours} work hours</small>
                    </td>
                    <td>{item.department}</td>
                    <td>{item.corridor_id}</td>
                    <td>
                      <strong>{formatNumber(item.model_risk_probability_percent)}%</strong>
                    </td>
                    <td>
                      <span className={priorityClass(item.priority_level)}>
                        {item.priority_level}
                      </span>
                    </td>
                    <td>
                      <span className={statusClass(item.status)}>
                        {item.status}
                      </span>
                    </td>
                    <td>{formatDate(item.deadline)}</td>
                    <td>
                      <button
                        className="view-button"
                        onClick={() => setSelected(item)}
                      >
                        View
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        <div className="maintenance-pagination">
          <div>
            Showing {items.length} of {pagination?.total_records ?? 0} matching records
          </div>

          <div className="pagination-controls">
            <label>
              Rows
              <select
                value={pageSize}
                onChange={(e) => {
                  setPageSize(Number(e.target.value));
                  setPage(1);
                }}
              >
                <option value={10}>10</option>
                <option value={20}>20</option>
                <option value={50}>50</option>
                <option value={100}>100</option>
              </select>
            </label>

            <button
              className="secondary-button"
              disabled={!pagination?.has_previous || loading}
              onClick={() => setPage((current) => Math.max(1, current - 1))}
            >
              Previous
            </button>

            <span className="page-number">
              {pagination ? `${pagination.page} / ${pagination.total_pages}` : '—'}
            </span>

            <button
              className="secondary-button"
              disabled={!pagination?.has_next || loading}
              onClick={() => setPage((current) => current + 1)}
            >
              Next
            </button>
          </div>
        </div>
      </div>

      {selected && (
        <div className="request-detail-overlay" onClick={() => setSelected(null)}>
          <section
            className="request-detail-panel"
            role="dialog"
            aria-modal="true"
            aria-label={`Details for ${selected.request_id}`}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="detail-heading">
              <div>
                <div className="section-eyebrow">REQUEST DETAILS</div>
                <h2>{selected.request_id}</h2>
                <p>{selected.maintenance_type}</p>
              </div>
              <button
                className="secondary-button"
                onClick={() => setSelected(null)}
              >
                Close
              </button>
            </div>

            <div className="detail-grid">
              <div><span>Asset ID</span><strong>{selected.asset_id}</strong></div>
              <div><span>Record ID</span><strong>{selected.record_id}</strong></div>
              <div><span>Department</span><strong>{selected.department}</strong></div>
              <div><span>Corridor</span><strong>{selected.corridor_id}</strong></div>
              <div><span>Location</span><strong>KM {formatNumber(selected.location_km)}</strong></div>
              <div><span>Required duration</span><strong>{selected.required_duration_hours} hours</strong></div>
              <div><span>Team size</span><strong>{selected.team_size_required}</strong></div>
              <div><span>Priority score</span><strong>{formatNumber(selected.operational_priority_score)}</strong></div>
              <div><span>Model risk probability</span><strong>{formatNumber(selected.model_risk_probability_percent)}%</strong></div>
              <div><span>Equipment condition score</span><strong>{formatNumber(selected.equipment_condition_score)}</strong></div>
              <div><span>Earliest start</span><strong>{formatDate(selected.earliest_start)}</strong></div>
              <div><span>Deadline</span><strong>{formatDate(selected.deadline)}</strong></div>
              <div><span>Priority</span><strong>{selected.priority_level}</strong></div>
              <div><span>Status</span><strong>{selected.status}</strong></div>
              <div><span>Data classification</span><strong>{selected.is_simulated ? 'Simulated' : 'Not marked simulated'}</strong></div>
            </div>

            <p className="detail-disclaimer">
              This request is part of a planning prototype. It is not an
              authorization to carry out railway maintenance work.
            </p>
          </section>
        </div>
      )}
    </section>
  );
}
