
export default function Topbar() {
  return (
    <header className="topbar">
      <div>
        <div className="topbar-eyebrow">RAILWAY OPERATIONS</div>
        <div className="topbar-title">Maintenance Intelligence</div>
      </div>

      <div className="topbar-right">
        <div className="environment">
          <span className="status-dot" />
          DEVELOPMENT ENVIRONMENT
        </div>
        <div className="topbar-date">
          SYSTEM DASHBOARD
        </div>
      </div>
    </header>
  )
}
