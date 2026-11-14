import { Link } from 'react-router-dom'

export default function NotFound() {
  return (
    <section className="page-section" style={{ textAlign: 'center', padding: '4rem 1rem' }}>
      <div className="eyebrow" style={{ color: '#E11D48', marginBottom: '0.5rem' }}>ERROR 404</div>
      <h1 style={{ fontSize: '2rem', marginBottom: '0.75rem' }}>Page Not Found</h1>
      <p style={{ color: '#6B7280', marginBottom: '1.5rem', maxWidth: '400px', margin: '0 auto 1.5rem auto' }}>
        The route you requested does not exist in the RailSetu AI navigation system.
      </p>
      <Link to="/" className="primary-button" style={{ display: 'inline-block', textDecoration: 'none' }}>
        Return to Overview
      </Link>
    </section>
  )
}
