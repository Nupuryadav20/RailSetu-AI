
import { useState } from 'react'
import { dashboardApi } from '../services/api'

const fields = [
    ['region', 'Region', 'text'],
    ['season', 'Season', 'text'],
    ['train_type', 'Train Type', 'text'],
    ['train_age_years', 'Train Age (years)', 'number'],
    ['average_speed_kmph', 'Average Speed (km/h)', 'number'],
    ['distance_travelled_km', 'Distance Travelled (km)', 'number'],
    ['track_temperature_c', 'Track Temperature (°C)', 'number'],
    ['rail_wear_mm', 'Rail Wear (mm)', 'number'],
    ['track_vibration_level', 'Track Vibration Level', 'number'],
    ['ballast_condition', 'Ballast Condition', 'text'],
    ['track_curvature_degree', 'Track Curvature (degree)', 'number'],
    ['ambient_temperature_c', 'Ambient Temperature (°C)', 'number'],
    ['humidity_percent', 'Humidity (%)', 'number'],
    ['rainfall_mm', 'Rainfall (mm)', 'number'],
    ['wind_speed_kmph', 'Wind Speed (km/h)', 'number'],
    ['wheel_wear_percent', 'Wheel Wear (%)', 'number'],
    ['axle_temperature_c', 'Axle Temperature (°C)', 'number'],
    ['brake_pressure_psi', 'Brake Pressure (PSI)', 'number'],
    ['brake_pad_wear_percent', 'Brake Pad Wear (%)', 'number'],
    ['bearing_temperature_c', 'Bearing Temperature (°C)', 'number'],
    ['battery_voltage', 'Battery Voltage', 'number'],
    ['traction_motor_temp_c', 'Traction Motor Temperature (°C)', 'number'],
    ['signal_system_status', 'Signal System Status', 'text'],
    ['power_consumption_kw', 'Power Consumption (kW)', 'number'],
    ['load_factor_percent', 'Load Factor (%)', 'number'],
    ['daily_trips', 'Daily Trips', 'number'],
    ['delay_minutes', 'Delay (minutes)', 'number'],
    ['last_maintenance_days', 'Days Since Last Maintenance', 'number'],
    ['inspection_score', 'Inspection Score', 'number'],
    ['sensor_health_index', 'Sensor Health Index', 'number'],
] as const

const numericFields = new Set<string>(
    fields.filter((field) => field[2] === 'number').map((field) => field[0])
)

export default function PredictiveMaintenance() {
    const [values, setValues] = useState<Record<string, string>>(
    Object.fromEntries(fields.map(([key]) => [key, '']))
    )
    const [result, setResult] = useState<unknown>(null)
    const [error, setError] = useState('')
    const [loading, setLoading] = useState(false)

    function updateField(key: string, value: string) {
    setValues((previous) => ({ ...previous, [key]: value }))
    }

    async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setError('')
    setResult(null)

    const record: Record<string, string | number> = {}

    for (const [key, label] of fields) {
        const value = values[key]?.trim()

        if (!value) {
            setError(`Please enter ${label}.`)
            return
        }

        if (numericFields.has(key)) {
        const numberValue = Number(value)

        if (!Number.isFinite(numberValue)) {
            setError(`${label} must be a valid number.`)
            return
        }

        record[key] = numberValue
      } else {
        record[key] = value
      }
    }

    setLoading(true)

    try {
      const response = await dashboardApi.predict([record])
      setResult(response.data)
    } catch (err: any) {
      setError(
        err?.response?.data?.detail
          ? JSON.stringify(err.response.data.detail)
          : err?.message || 'Prediction request failed. Check that the backend is running.'
      )
    } finally {
      setLoading(false)
    }
  }

  function resetForm() {
    setValues(Object.fromEntries(fields.map(([key]) => [key, ''])))
    setResult(null)
    setError('')
  }

  return (
    <section className="page-section">
      <div className="page-heading">
        <div>
          <div className="eyebrow">MACHINE LEARNING / INFERENCE</div>
          <h1>Predictive Maintenance</h1>
          <p>Submit asset and operating conditions to generate a model prediction.</p>
        </div>
        <div className="data-label">MODEL-BASED PREDICTION</div>
      </div>

      <div className="panel">
        <div className="panel-heading">
          <div>
            <h2>Asset condition inputs</h2>
            <p>All 30 features required by the prediction API must be provided.</p>
          </div>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="prediction-grid">
            {fields.map(([key, label, type]) => (
              <label className="prediction-field" key={key}>
                <span>{label}</span>
                <input
                  type={type}
                  step={type === 'number' ? 'any' : undefined}
                  value={values[key]}
                  onChange={(event) => updateField(key, event.target.value)}
                  placeholder={`Enter ${label.toLowerCase()}`}
                  required
                />
              </label>
            ))}
          </div>

          {error && <div className="error-message">{error}</div>}

          <div className="prediction-actions">
            <button type="button" className="secondary-button" onClick={resetForm}>
              Clear form
            </button>
            <button type="submit" className="primary-button" disabled={loading}>
              {loading ? 'Generating prediction...' : 'Generate prediction'}
            </button>
          </div>
        </form>
      </div>

      {result !== null && (
        <div className="panel prediction-result">
          <div className="eyebrow">API RESPONSE</div>
          <h2>Prediction result</h2>
          <p>Raw response returned by the trained model API:</p>
          <pre>{JSON.stringify(result, null, 2)}</pre>
        </div>
      )}

      <p className="disclaimer">
        Research prototype using the project's dataset. Predictions are experimental
        decision-support outputs and must not be used as authorization for railway
        maintenance or operational decisions.
      </p>
    </section>
  )
}
