
import { NavLink } from 'react-router-dom'

const navigation = [
  { label: 'Overview', path: '/' },
  { label: 'Predictive Maintenance', path: '/prediction' },
  { label: 'Maintenance Requests', path: '/maintenance' },
  { label: 'Block Planning', path: '/planning' },
  { label: 'Model Insights', path: '/insights' },
]

export default function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-name">RAILSETU<span> AI</span></div>
        <div className="brand-caption">RAILWAY INTELLIGENCE SYSTEM</div>
      </div>

      <div className="nav-section-label">WORKSPACE</div>

      <nav className="nav-list">
        {navigation.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            end={item.path === '/'}
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
        <NavLink to="/privacy" className="nav-link">
          Privacy & Policy
        </NavLink>
        <div className="system-version">RAILSETU AI · VERSION 1.0.0</div>
      </div>
    </aside>
  )
}
