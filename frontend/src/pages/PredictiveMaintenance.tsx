import { useState, useEffect } from 'react'
import { dashboardApi } from '../services/api'

interface DiagnosticRow {
  feature: string
  value: any
  train_mean: number | null
  train_std: number | null
  z_score: number | null
  percentile_rank: number | null
  class_0_mean: number | null
  class_1_mean: number | null
  shap_contribution: number
}

interface PredictionResult {
  model: string
  model_version: string
  is_calibrated: boolean
  timestamp: string
  threshold: number
  threshold_basis: string
  prediction_type: string
  honest_note: string
  concept_note: string
  results: Array<{
    maintenance_probability_percent: number
    calibrated_probability_percent: number
    predicted_class: number
    threshold: number
    decision: string
    warnings: string[]
    top_shap_contributors: Array<{ feature: string; shap_value: number }>
    diagnostics: DiagnosticRow[]
  }>
}

interface HistoryItem {
  id: number
  timestamp: string
  model_name: string
  model_version: string
  probability: number
  probability_percent: number
  threshold: number
  predicted_class: number
  top_shap: Array<{ feature: string; shap_value: number }>
  warnings: string[]
}

const CATEGORICAL_OPTIONS: Record<string, string[]> = {
  region: ['Southern Railway', 'Northern Railway', 'Eastern Railway', 'Western Railway', 'Central Railway'],
  season: ['Summer', 'Monsoon', 'Winter', 'Spring'],
  train_type: ['Freight', 'Passenger', 'Express', 'High-Speed'],
  ballast_condition: ['good', 'fair', 'poor'],
  signal_system_status: ['operational', 'warning', 'fault'],
}

const SAMPLE_NORMAL: Record<string, string | number> = {
  region: 'Northern Railway',
  season: 'Winter',
  train_type: 'Express',
  train_age_years: 8,
  average_speed_kmph: 85,
  distance_travelled_km: 145000,
  track_temperature_c: 24,
  rail_wear_mm: 3.2,
  track_vibration_level: 1.8,
  ballast_condition: 'good',
  track_curvature_degree: 1.5,
  ambient_temperature_c: 21,
  humidity_percent: 62,
  rainfall_mm: 0,
  wind_speed_kmph: 14,
  wheel_wear_percent: 28,
  axle_temperature_c: 52,
  brake_pressure_psi: 95,
  brake_pad_wear_percent: 32,
  bearing_temperature_c: 58,
  battery_voltage: 24.2,
  traction_motor_temp_c: 84,
  signal_system_status: 'operational',
  power_consumption_kw: 1250,
  load_factor_percent: 74,
  daily_trips: 4,
  delay_minutes: 5,
  last_maintenance_days: 28,
  inspection_score: 88,
  sensor_health_index: 92,
}

const SAMPLE_CRITICAL: Record<string, string | number> = {
  ...SAMPLE_NORMAL,
  rail_wear_mm: 14.8,
  track_vibration_level: 6.4,
  ballast_condition: 'poor',
  wheel_wear_percent: 86,
  axle_temperature_c: 98,
  bearing_temperature_c: 112,
  brake_pad_wear_percent: 89,
  signal_system_status: 'warning',
  inspection_score: 34,
  sensor_health_index: 41,
  last_maintenance_days: 145,
}

export default function PredictiveMaintenance() {
  const [form, setForm] = useState<Record<string, string | number>>(SAMPLE_NORMAL)
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [history, setHistory] = useState<HistoryItem[]>([])
  const [classFilter, setClassFilter] = useState<string>('all')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const fetchHistory = async () => {
    try {
      const params: any = { limit: 50 }
      if (classFilter !== 'all') {
        params.predicted_class = Number(classFilter)
      }
      const res = await dashboardApi.getPredictions(params)
      if (Array.isArray(res.data)) setHistory(res.data)
    } catch {
      // Non-fatal
    }
  }

  useEffect(() => {
    fetchHistory()
  }, [classFilter])

  const handleChange = (key: string, value: string | number) => {
    setForm((prev) => ({ ...prev, [key]: value }))
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    setResult(null)
    setLoading(true)

    const payloadRecord: Record<string, unknown> = {}
    for (const [k, v] of Object.entries(form)) {
      payloadRecord[k] = CATEGORICAL_OPTIONS[k] ? String(v) : Number(v)
    }

    try {
      const res = await dashboardApi.predict([payloadRecord])
      setResult(res.data as PredictionResult)
      fetchHistory()
    } catch (err: any) {
      const msg = err?.response?.data?.detail?.message || err?.response?.data?.detail || err.message
      setError(typeof msg === 'string' ? msg : JSON.stringify(msg))
    } finally {
      setLoading(false)
    }
  }

  const singleResult = result?.results?.[0]

  return (
    <section className="page-section">
      <div className="page-heading">
        <div>
          <div className="eyebrow">ML INFERENCE & DECISION SUPPORT (V5 ACTIVE)</div>
          <h1>Predictive Maintenance (Concept A)</h1>
          <p>
            Generate real-time maintenance predictions with feature z-scores, percentile ranks, and SHAP attribution.
          </p>
        </div>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <button
            type="button"
            className="secondary-button"
            onClick={() => setForm(SAMPLE_NORMAL)}
          >
            Load Normal Sample
          </button>
          <button
            type="button"
            className="secondary-button"
            onClick={() => setForm(SAMPLE_CRITICAL)}
          >
            Load High-Wear Sample
          </button>
        </div>
      </div>

      {/* Concept Separation Alert */}
      <div
        style={{
          background: '#EFF6FF',
          border: '1px solid #BFDBFE',
          borderRadius: '12px',
          padding: '1rem 1.25rem',
          marginBottom: '1.5rem',
          fontSize: '0.875rem',
          color: '#1E3A8A',
        }}
      >
        <strong>Concept Distinction Note:</strong> This form calculates <strong>Concept A (ML Prediction Probability)</strong>.
        It is strictly independent of <strong>Concept B (Dataset Risk Score)</strong> which was excluded from features to prevent leakage,
        and <strong>Concept C (Operational Priority Score)</strong> which blends physical wear with corridor criticality.
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1.3fr) minmax(0, 1fr)', gap: '1.5rem' }}>
        {/* Form Panel */}
        <div className="panel">
          <div className="panel-heading">
            <div>
              <h2>Telemetry Inputs (30 Evaluated Features)</h2>
              <p>Values are checked against training distribution percentiles in real time.</p>
            </div>
          </div>

          <form onSubmit={handleSubmit}>
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))',
                gap: '0.75rem',
                maxHeight: '480px',
                overflowY: 'auto',
                paddingRight: '0.5rem',
              }}
            >
              {Object.keys(SAMPLE_NORMAL).map((key) => {
                const isCat = CATEGORICAL_OPTIONS[key] !== undefined
                const label = key.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())

                return (
                  <div key={key} style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                    <label style={{ fontSize: '0.75rem', color: '#4B5563', fontWeight: 500 }}>
                      {label}
                    </label>
                    {isCat ? (
                      <select
                        value={form[key] ?? ''}
                        onChange={(e) => handleChange(key, e.target.value)}
                        style={{
                          padding: '0.45rem 0.6rem',
                          borderRadius: '8px',
                          border: '1px solid #D1D5DB',
                          fontSize: '0.85rem',
                          background: '#fff',
                        }}
                      >
                        {CATEGORICAL_OPTIONS[key].map((opt) => (
                          <option key={opt} value={opt}>
                            {opt}
                          </option>
                        ))}
                      </select>
                    ) : (
                      <input
                        type="number"
                        step="any"
                        value={form[key] ?? ''}
                        onChange={(e) => handleChange(key, e.target.value)}
                        required
                        style={{
                          padding: '0.45rem 0.6rem',
                          borderRadius: '8px',
                          border: '1px solid #D1D5DB',
                          fontSize: '0.85rem',
                          background: '#fff',
                        }}
                      />
                    )}
                  </div>
                )
              })}
            </div>

            {error && (
              <div
                style={{
                  background: '#FEE2E2',
                  border: '1px solid #FCA5A5',
                  color: '#991B1B',
                  borderRadius: '8px',
                  padding: '0.75rem',
                  marginTop: '1rem',
                  fontSize: '0.85rem',
                }}
              >
                {error}
              </div>
            )}

            <div style={{ display: 'flex', gap: '0.75rem', marginTop: '1.25rem' }}>
              <button type="submit" className="primary-button" disabled={loading}>
                {loading ? 'Evaluating Model...' : 'Run ML Prediction'}
              </button>
            </div>
          </form>
        </div>

        {/* Results Panel */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          {singleResult ? (
            <div className="panel" style={{ borderLeft: singleResult.predicted_class === 1 ? '4px solid #E11D48' : '4px solid #2E9E3E' }}>
              <div className="eyebrow">PREDICTION OUTPUT</div>
              <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.75rem', marginTop: '0.5rem' }}>
                <span style={{ fontSize: '2.5rem', fontWeight: 300, color: '#111827' }}>
                  {singleResult.maintenance_probability_percent}%
                </span>
                <span
                  style={{
                    padding: '0.25rem 0.6rem',
                    borderRadius: '999px',
                    fontSize: '0.75rem',
                    fontWeight: 600,
                    background: singleResult.predicted_class === 1 ? '#FFE4E6' : '#DCFCE7',
                    color: singleResult.predicted_class === 1 ? '#9F1239' : '#166534',
                  }}
                >
                  {singleResult.decision}
                </span>
              </div>

              <div style={{ fontSize: '0.8rem', color: '#6B7280', marginTop: '0.75rem', lineHeight: '1.5' }}>
                <div><strong>Calibrated Probability:</strong> {singleResult.calibrated_probability_percent}% (Isotonic)</div>
                <div><strong>Model:</strong> {result?.model} ({result?.model_version})</div>
                <div><strong>Operational Threshold:</strong> {singleResult.threshold} ({result?.threshold_basis})</div>
                <div><strong>Timestamp:</strong> {new Date(result?.timestamp || '').toLocaleTimeString()}</div>
              </div>

              {/* Warnings */}
              {singleResult.warnings && singleResult.warnings.length > 0 && (
                <div style={{ marginTop: '0.75rem', padding: '0.6rem', background: '#FEF2F2', borderRadius: '6px', border: '1px solid #FECACA', fontSize: '0.75rem', color: '#991B1B' }}>
                  <strong>Distribution Alerts:</strong>
                  <ul style={{ margin: '0.25rem 0 0 1rem', padding: 0 }}>
                    {singleResult.warnings.map((w, i) => (
                      <li key={i}>{w}</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* SHAP Contributors */}
              {singleResult.top_shap_contributors && singleResult.top_shap_contributors.length > 0 && (
                <div style={{ marginTop: '1rem', paddingTop: '0.75rem', borderTop: '1px solid #E5E7EB' }}>
                  <div style={{ fontSize: '0.75rem', fontWeight: 600, color: '#374151', textTransform: 'uppercase', marginBottom: '0.4rem' }}>
                    Top Local SHAP Contributors
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
                    {singleResult.top_shap_contributors.map((c) => (
                      <div
                        key={c.feature}
                        style={{
                          display: 'flex',
                          justifyContent: 'space-between',
                          fontSize: '0.8rem',
                          background: '#F9FAFB',
                          padding: '0.35rem 0.5rem',
                          borderRadius: '6px',
                        }}
                      >
                        <span style={{ color: '#4B5563' }}>{c.feature.replace(/_/g, ' ')}</span>
                        <span style={{ fontWeight: 600, color: c.shap_value >= 0 ? '#E11D48' : '#2E9E3E' }}>
                          {c.shap_value >= 0 ? `+${c.shap_value.toFixed(4)}` : c.shap_value.toFixed(4)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Honest Model Note */}
              <div
                style={{
                  marginTop: '1rem',
                  padding: '0.75rem',
                  borderRadius: '8px',
                  background: '#FEF3C7',
                  border: '1px solid #FDE68A',
                  fontSize: '0.75rem',
                  color: '#92400E',
                }}
              >
                ⚠️ <strong>Limitation:</strong> {result?.honest_note}
              </div>
            </div>
          ) : (
            <div className="panel" style={{ textAlign: 'center', padding: '3rem 1rem', color: '#9CA3AF' }}>
              <div style={{ fontSize: '1.1rem', marginBottom: '0.5rem' }}>Awaiting Telemetry Input</div>
              <p style={{ fontSize: '0.85rem' }}>Fill or load sample inputs and click "Run ML Prediction".</p>
            </div>
          )}

          {/* Database History Panel */}
          <div className="panel">
            <div className="panel-heading" style={{ marginBottom: '0.75rem' }}>
              <div>
                <h3 style={{ fontSize: '0.95rem' }}>Persistent Database Predictions</h3>
                <p style={{ fontSize: '0.75rem' }}>Saved to SQLite (<code>predictions</code> table)</p>
              </div>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <select
                  value={classFilter}
                  onChange={(e) => setClassFilter(e.target.value)}
                  style={{ fontSize: '0.75rem', padding: '0.2rem 0.4rem', borderRadius: '4px', border: '1px solid #D1D5DB' }}
                >
                  <option value="all">All Classes</option>
                  <option value="1">Class 1 (Flagged)</option>
                  <option value="0">Class 0 (Normal)</option>
                </select>
                <a
                  href={dashboardApi.getPredictionsDownloadUrl()}
                  target="_blank"
                  rel="noreferrer"
                  style={{
                    fontSize: '0.75rem',
                    padding: '0.2rem 0.5rem',
                    borderRadius: '4px',
                    background: '#2563EB',
                    color: '#FFF',
                    textDecoration: 'none',
                    display: 'flex',
                    alignItems: 'center',
                  }}
                >
                  Download CSV
                </a>
              </div>
            </div>

            {history.length === 0 ? (
              <div style={{ fontSize: '0.8rem', color: '#9CA3AF' }}>No prediction records found in database.</div>
            ) : (
              <div style={{ maxHeight: '180px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
                {history.slice(0, 6).map((h) => (
                  <div
                    key={h.id}
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      fontSize: '0.78rem',
                      padding: '0.35rem 0.5rem',
                      background: '#F9FAFB',
                      borderRadius: '6px',
                    }}
                  >
                    <span style={{ color: '#6B7280' }}>
                      #{h.id} • {new Date(h.timestamp).toLocaleTimeString()} ({h.model_version})
                    </span>
                    <span
                      style={{
                        fontWeight: 600,
                        color: h.predicted_class === 1 ? '#E11D48' : '#2E9E3E',
                      }}
                    >
                      {h.probability_percent}% → Class {h.predicted_class}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Feature Diagnostics Detailed Table */}
      {singleResult && singleResult.diagnostics && (
        <div className="panel" style={{ marginTop: '1.5rem' }}>
          <div className="panel-heading">
            <div>
              <h2>Per-Feature Distribution Diagnostics & Anomaly Scores</h2>
              <p>Compares inputs with training set statistics, z-score deviations, and class conditional means.</p>
            </div>
          </div>
          <div style={{ overflowX: 'auto', maxHeight: '400px' }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Feature</th>
                  <th>Input Value</th>
                  <th>Train Mean ± Std</th>
                  <th>Z-Score</th>
                  <th>Percentile</th>
                  <th>Class 0 Mean</th>
                  <th>Class 1 Mean</th>
                  <th>SHAP Impact</th>
                </tr>
              </thead>
              <tbody>
                {singleResult.diagnostics.map((d) => {
                  const isHighZ = d.z_score !== null && Math.abs(d.z_score) > 2.0
                  return (
                    <tr key={d.feature}>
                      <td><strong>{d.feature.replace(/_/g, ' ')}</strong></td>
                      <td>{String(d.value)}</td>
                      <td>
                        {d.train_mean !== null ? `${d.train_mean.toFixed(2)} ± ${d.train_std?.toFixed(2)}` : 'Categorical'}
                      </td>
                      <td>
                        {d.z_score !== null ? (
                          <span
                            style={{
                              padding: '0.15rem 0.4rem',
                              borderRadius: '4px',
                              fontSize: '0.75rem',
                              fontWeight: 600,
                              background: isHighZ ? '#FEE2E2' : '#F3F4F6',
                              color: isHighZ ? '#991B1B' : '#374151',
                            }}
                          >
                            {d.z_score > 0 ? `+${d.z_score.toFixed(2)}` : d.z_score.toFixed(2)}
                          </span>
                        ) : '—'}
                      </td>
                      <td>{d.percentile_rank !== null ? `${d.percentile_rank.toFixed(1)}%` : '—'}</td>
                      <td>{d.class_0_mean !== null ? d.class_0_mean.toFixed(2) : '—'}</td>
                      <td>{d.class_1_mean !== null ? d.class_1_mean.toFixed(2) : '—'}</td>
                      <td style={{ fontWeight: 600, color: d.shap_contribution >= 0 ? '#E11D48' : '#2E9E3E' }}>
                        {d.shap_contribution !== 0 ? (d.shap_contribution >= 0 ? `+${d.shap_contribution.toFixed(4)}` : d.shap_contribution.toFixed(4)) : '0.0000'}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </section>
  )
}
