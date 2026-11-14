import { NavLink } from 'react-router-dom'

const navigation = [
  { label: 'Overview', path: '/' },
  { label: 'Predictive Maintenance', path: '/prediction' },
  { label: 'Maintenance Requests', path: '/maintenance' },
  { label: 'Block Planning', path: '/planning' },
  { label: 'Model Insights', path: '/insights' },
]

interface SidebarProps {
  mobileOpen?: boolean
  onClose?: () => void
}

export default function Sidebar({ mobileOpen, onClose }: SidebarProps) {
  const handleNavClick = () => {
    if (onClose) onClose()
  }

  return (
    <aside className={`sidebar ${mobileOpen ? 'open' : ''}`}>
      <div className="brand" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <div className="brand-name">RAILSETU<span> AI</span></div>
          <div className="brand-caption">RAILWAY INTELLIGENCE SYSTEM</div>
        </div>
        {mobileOpen && (
          <button
            type="button"
            className="mobile-close-btn"
            onClick={onClose}
            aria-label="Close sidebar"
            style={{
              background: 'none',
              border: 'none',
              fontSize: '1.25rem',
              color: '#64748B',
              cursor: 'pointer',
              padding: '0 4px',
            }}
          >
            ✕
          </button>
        )}
      </div>

      <div className="nav-section-label">WORKSPACE</div>

      <nav className="nav-list">
        {navigation.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            end={item.path === '/'}
            onClick={handleNavClick}
            className={({ isActive }) =>
              `nav-link ${isActive ? 'active' : ''}`
            }
          >
            {item.label}
          </NavLink>
        ))}
      </nav>

      <div className="sidebar-bottom">
        <div className="nav-section-label">SYSTEM</div>
        <NavLink to="/privacy" className="nav-link" onClick={handleNavClick}>
          Privacy & Policy
        </NavLink>
        <div className="system-version">RAILSETU AI · VERSION 1.0.0</div>
      </div>
    </aside>
  )
}
