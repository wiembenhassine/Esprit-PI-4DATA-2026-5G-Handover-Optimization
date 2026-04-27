import { Outlet, NavLink, useNavigate, useLocation } from 'react-router-dom'
import { useAuth } from '../AuthContext'
import { useState, useEffect } from 'react'
import { getHealth } from '../api'

const ROLE_COLOR = {
  admin:           '#ff6b35',
  data_scientist:  '#00ff9d',
  network_operator:'#00d4ff',
}

const NAV = [
  { to:'/',            icon:'◈', label:'Overview',      section:'MONITORING' },
  { to:'/predict',     icon:'▶', label:'Predict',       section:null },
  { to:'/history',     icon:'≡', label:'History',       section:null },
  { to:'/monitoring',  icon:'⚠', label:'Drift & Alerts',section:null, badge:true },
  { to:'/mlops',       icon:'⚙', label:'MLOps',         section:'MANAGEMENT', scientist:true },
]

export default function Layout() {
  const { user, logout, isScientist } = useAuth()
  const navigate  = useNavigate()
  const location  = useLocation()
  const [health, setHealth] = useState('checking')

  useEffect(() => {
    getHealth()
      .then(d => setHealth(d?.platform_status || 'healthy'))
      .catch(() => setHealth('degraded'))
  }, [])

  function handleLogout() {
    logout()
    navigate('/login', { replace: true })
  }

  const titles = {
    '/':          'Platform Overview',
    '/predict':   'Run Prediction',
    '/history':   'Prediction History',
    '/monitoring':'Drift & Alerts',
    '/mlops':     'Model Registry & MLOps',
  }
  const title = titles[location.pathname] || 'Platform'

  const initials = user?.username?.slice(0, 2).toUpperCase() || '??'
  const roleColor = ROLE_COLOR[user?.role] || '#00d4ff'

  return (
    <div className="app-shell">
      {/* ── Sidebar ── */}
      <aside className="sidebar">
        <div className="sidebar-logo">
          <div className="brand">5G·HANDOVER</div>
          <div className="sub">AI Optimization Platform</div>
        </div>

        <nav className="sidebar-nav">
          {NAV.map((item, idx) => {
            if (item.scientist && !isScientist) return null
            const els = []
            if (item.section) {
              els.push(<div key={`sec-${idx}`} className="nav-section">{item.section}</div>)
            }
            els.push(
              <NavLink
                key={item.to}
                to={item.to}
                end={item.to === '/'}
                className={({ isActive }) => `nav-item${isActive ? ' active' : ''}`}
              >
                <span className="nav-icon">{item.icon}</span>
                <span>{item.label}</span>
                {item.badge && <span className="nav-badge">!</span>}
              </NavLink>
            )
            return els
          })}
        </nav>

        <div className="sidebar-footer">
          {/* Health indicator */}
          <div style={{ display:'flex', alignItems:'center', gap:6, marginBottom:10, fontSize:10, color:'#7a8fa6' }}>
            <span className={`status-dot${health === 'healthy' ? '' : ' warn'}`} />
            <span>Platform {health}</span>
          </div>
          {/* User row */}
          <div className="user-row">
            <div className="user-avatar" style={{ borderColor:`${roleColor}44`, background:`${roleColor}18`, color:roleColor }}>
              {initials}
            </div>
            <div style={{ flex:1, minWidth:0 }}>
              <div className="user-name">{user?.username}</div>
              <div className="user-role" style={{ color:roleColor }}>
                {user?.role?.replace(/_/g,' ')}
              </div>
            </div>
            <button className="logout-btn" title="Sign out" onClick={handleLogout}>⏻</button>
          </div>
        </div>
      </aside>

      {/* ── Main ── */}
      <div className="main-area">
        <div className="topbar">
          <span className="topbar-title">{title}</span>
          <span className="chip chip-blue">DSO1→DSO2→DSO3→DSO4</span>
          <div style={{ display:'flex', alignItems:'center', gap:6, fontSize:10, color:'#00ff9d' }}>
            <span className="live-dot" />
            <span>LIVE</span>
          </div>
        </div>
        <div className="page-content">
          <Outlet />
        </div>
      </div>
    </div>
  )
}
