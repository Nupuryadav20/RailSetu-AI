interface TopbarProps {
  onToggleMenu?: () => void
}

export default function Topbar({ onToggleMenu }: TopbarProps) {
  return (
    <header className="topbar">
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
        <button
          type="button"
          className="mobile-menu-btn"
          onClick={onToggleMenu}
          aria-label="Toggle navigation menu"
          style={{
            background: '#FFFFFF',
            border: '1px solid #CBD5E1',
            borderRadius: '6px',
            padding: '6px 10px',
            cursor: 'pointer',
            fontSize: '1rem',
            color: '#0F172A',
            display: 'none',
          }}
        >
          ☰
        </button>
        <div>
          <div className="topbar-eyebrow">RAILWAY OPERATIONS</div>
          <div className="topbar-title">Maintenance Intelligence</div>
        </div>
      </div>

      <div className="topbar-right">
        <div className="environment" title="Development Environment">
          <span className="status-dot" />
          <span className="env-label">DEVELOPMENT ENVIRONMENT</span>
        </div>
        <div className="topbar-date">
          SYSTEM DASHBOARD
        </div>
      </div>
    </header>
  )
}
