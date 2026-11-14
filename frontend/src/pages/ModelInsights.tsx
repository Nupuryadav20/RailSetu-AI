
import { useEffect, useState } from 'react'
import { dashboardApi } from '../services/api'

type Row = Record<string, unknown>

function rowsFrom(value: unknown): Row[] {
  if (Array.isArray(value)) return value.filter(x => x && typeof x === 'object') as Row[]
  if (value && typeof value === 'object') {
    const obj = value as Record<string, unknown>
    for (const key of ['items', 'records', 'results', 'data', 'models', 'features', 'checks']) {
      if (Array.isArray(obj[key])) return rowsFrom(obj[key])
    }
    return [obj]
  }
  return []
}

function formatValue(value: unknown): string {
  if (value === null || value === undefined) return '—'
  if (typeof value === 'boolean') return value ? 'Yes' : 'No'
  if (typeof value === 'number') return Number.isInteger(value) ? value.toLocaleString() : value.toFixed(4)
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}

function titleCase(key: string): string {
  return key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
}

function DataTable({ data, empty }: { data: Row[]; empty: string }) {
  if (!data.length) return <p className="muted">{empty}</p>
  const columns = Object.keys(data[0]).filter(key => !key.startsWith('_'))
  return (
    <div className="table-wrap">
      <table className="data-table">
        <thead><tr>{columns.map(key => <th key={key}>{titleCase(key)}</th>)}</tr></thead>
        <tbody>
          {data.map((row, index) => (
            <tr key={index}>
              {columns.map(key => <td key={key}>{formatValue(row[key])}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default function ModelInsights() {
  const [comparison, setComparison] = useState<Row[]>([])
  const [bestModel, setBestModel] = useState<Row | null>(null)
  const [features, setFeatures] = useState<Row[]>([])
  const [validation, setValidation] = useState<Row[]>([])
  const [loading, setLoading] = useState(true)
  const [errors, setErrors] = useState<string[]>([])

  useEffect(() => {
    let active = true

    async function loadInsights() {
      setLoading(true)
      const results = await Promise.allSettled([
        dashboardApi.getModelComparison(),
        dashboardApi.getBestModel(),
        dashboardApi.getFeatureImportance(),
        dashboardApi.getValidation(),
      ])

      if (!active) return

      const failed: string[] = []
      results.forEach((result, index) => {
        if (result.status === 'rejected') {
          failed.push(['Model comparison', 'Best model', 'Feature importance', 'Validation'][index])
        }
      })

      if (results[0].status === 'fulfilled') setComparison(rowsFrom(results[0].value.data))
      if (results[1].status === 'fulfilled') {
        const data = results[1].value.data
        setBestModel(data && typeof data === 'object' && !Array.isArray(data)
          ? data as Row : rowsFrom(data)[0] ?? null)
      }
      if (results[2].status === 'fulfilled') setFeatures(rowsFrom(results[2].value.data))
      if (results[3].status === 'fulfilled') setValidation(rowsFrom(results[3].value.data))

      setErrors(failed)
      setLoading(false)
    }

    loadInsights()
    return () => { active = false }
  }, [])

  const validationPassed = validation.filter(row =>
    Object.entries(row).some(([key, value]) =>
      /pass|valid|success/i.test(key) && (value === true || String(value).toLowerCase() === 'passed')
    )
  ).length

  return (
    <section className="page-section">
      <div className="page-header">
        <div>
          <div className="eyebrow">MACHINE LEARNING / MODEL EVALUATION</div>
          <h1>Model Insights</h1>
          <p>Review model performance, influential features and planning validation results.</p>
        </div>
        <div className="data-source-label">EXPERIMENTAL MODEL OUTPUTS</div>
      </div>

      {loading && <div className="panel">Loading model insights from backend...</div>}

      {!loading && errors.length > 0 && (
        <div className="panel">
          <strong>Some model data could not be loaded.</strong>
          <p className="muted">Unavailable sections: {errors.join(', ')}. Other available sections are shown below.</p>
        </div>
      )}

      {!loading && (
        <>
          <div className="stat-grid">
            <article className="stat-card">
              <span>Models compared</span>
              <strong>{comparison.length}</strong>
            </article>
            <article className="stat-card">
              <span>Features returned</span>
              <strong>{features.length}</strong>
            </article>
            <article className="stat-card">
              <span>Validation checks</span>
              <strong>{validation.length}</strong>
            </article>
          </div>

          <section className="panel">
            <div className="section-heading">
              <div>
                <div className="eyebrow">MODEL SELECTION</div>
                <h2>Best performing model</h2>
              </div>
            </div>
            {bestModel
              ? <DataTable data={[bestModel]} empty="No best-model details returned." />
              : <p className="muted">Best model information is not available.</p>}
          </section>

          <section className="panel">
            <div className="section-heading">
              <div>
                <div className="eyebrow">PERFORMANCE EVALUATION</div>
                <h2>Model comparison</h2>
                <p className="muted">Metrics returned by the model evaluation API.</p>
              </div>
            </div>
            <DataTable data={comparison} empty="No model comparison records returned." />
          </section>

          <section className="panel">
            <div className="section-heading">
              <div>
                <div className="eyebrow">MODEL EXPLAINABILITY</div>
                <h2>Feature importance</h2>
                <p className="muted">Features ranked by the importance values supplied by the backend.</p>
              </div>
            </div>
            <DataTable data={features} empty="No feature importance records returned." />
          </section>

          <section className="panel">
            <div className="section-heading">
              <div>
                <div className="eyebrow">SCHEDULING ENGINE</div>
                <h2>Optimization validation</h2>
                <p className="muted">
                  {validation.length
                    ? `${validationPassed} checks identified as passed from the returned validation fields.`
                    : 'Validation results are not available.'}
                </p>
              </div>
            </div>
            <DataTable data={validation} empty="No validation records returned." />
          </section>

          <div className="disclaimer">
            Model outputs are experimental decision-support results based on the project's dataset.
            They are not certified railway safety predictions and must not be used as authorization
            for real-world maintenance or block operations.
          </div>
        </>
      )}
    </section>
  )
}
