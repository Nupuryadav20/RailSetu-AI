export default function PrivacyPolicy() {
  return (
    <section className="page-section">
      <div className="page-heading">
        <div>
          <div className="eyebrow">GOVERNANCE & COMPLIANCE</div>
          <h1>Privacy, Data Provenance & Safety Policy</h1>
          <p>Academic research prototype policies, synthetic data boundaries, and regulatory disclaimers.</p>
        </div>
      </div>

      <div className="panel" style={{ maxWidth: '840px', marginTop: '1.5rem' }}>
        <h2 style={{ fontSize: '1.25rem', marginBottom: '0.75rem' }}>1. Synthetic Data Notice</h2>
        <p style={{ lineHeight: '1.6', color: '#4B5563', marginBottom: '1.25rem' }}>
          RailSetu AI operates entirely on a synthetic benchmark dataset (100,000 records) modeled after railway equipment failure modes. 
          <strong> It does NOT connect to live Indian Railways telematics, CRIS (Centre for Railway Information Systems), or Ministry of Railways production databases.</strong> Any resemblance to active train schedules, locomotive serials, or operational personnel is purely coincidental.
        </p>

        <h2 style={{ fontSize: '1.25rem', marginBottom: '0.75rem' }}>2. Non-Certified Safety System</h2>
        <p style={{ lineHeight: '1.6', color: '#4B5563', marginBottom: '1.25rem' }}>
          This application is an academic decision-support prototype. The machine learning model exhibits modest discriminative ability (ROC-AUC ~0.55), reflecting the weak signal present in the synthetic dataset. 
          Outputs must never be used as sole authorization for safety-critical railway operations, track occupancy, or signaling overrides without certified engineering inspection.
        </p>

        <h2 style={{ fontSize: '1.25rem', marginBottom: '0.75rem' }}>3. Separation of Analytical Concepts</h2>
        <p style={{ lineHeight: '1.6', color: '#4B5563', marginBottom: '1.25rem' }}>
          The platform explicitly differentiates between:
          <br />• <strong>Concept A (ML Prediction Probability):</strong> Statistical likelihood ($0-100\%$) derived from 30 telemetry signals.
          <br />• <strong>Concept B (Dataset Risk Score):</strong> Synthetic label present in historical CSVs, strictly excluded from ML inputs to avoid target leakage.
          <br />• <strong>Concept C (Operational Priority Score):</strong> Rule-based composite ($0-100$) produced by the Priority Engine for CP-SAT block window scheduling.
        </p>

        <h2 style={{ fontSize: '1.25rem', marginBottom: '0.75rem' }}>4. Telemetry Privacy & Security</h2>
        <p style={{ lineHeight: '1.6', color: '#4B5563', marginBottom: '0.5rem' }}>
          The inference API does not store raw asset telemetry to persistent disk. Single-record predictions maintain an in-memory rolling ring-buffer (maximum 50 entries) of anonymized decision metadata solely for session inspection.
        </p>
      </div>
    </section>
  )
}
