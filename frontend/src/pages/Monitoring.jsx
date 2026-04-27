import { useEffect, useState } from 'react'
import { getDrift, getAlerts } from '../api'
import { RadarChart, Radar, PolarGrid, PolarAngleAxis, Tooltip, ResponsiveContainer } from 'recharts'

function DriftGauge({ name, psi, status }) {
  const max = 0.25
  const pct = Math.min((psi / max) * 100, 100)
  const color = status === 'ok' ? '#00ff9d' : status === 'warning' ? '#ffbe0b' : '#ff3860'
  return (
    <div className="drift-row">
      <span className="drift-name">{name}</span>
      <div className="drift-track">
        <div className="drift-fill" style={{ width:`${pct}%`, background:color }} />
      </div>
      <span className="drift-val" style={{ color }}>{psi?.toFixed(3)}</span>
      <span className="drift-status" style={{ color, fontSize:9 }}>{status?.toUpperCase()}</span>
    </div>
  )
}

const SEVERITY_ICON = { critical:'🔴', warning:'🟡', info:'🔵' }

export default function Monitoring() {
  const [drift,   setDrift]   = useState({})
  const [alerts,  setAlerts]  = useState([])
  const [loading, setLoading] = useState(true)
  const [filter,  setFilter]  = useState('')

  async function load() {
    setLoading(true)
    try {
      const [d, a] = await Promise.all([getDrift(), getAlerts()])
      setDrift(d || {})
      setAlerts(Array.isArray(a) ? a : [])
    } catch(e) {
      console.error(e)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  const radarData = Object.entries(drift)
    .filter(([,v]) => v?.psi !== undefined)
    .map(([k,v]) => ({ model: k.toUpperCase(), psi: v.psi || 0 }))

  const filteredAlerts = filter
    ? alerts.filter(a => a.severity === filter)
    : alerts

  const criticalCount = alerts.filter(a=>a.severity==='critical').length
  const warningCount  = alerts.filter(a=>a.severity==='warning').length

  return (
    <div className="fade-up">
      {/* Summary row */}
      <div className="grid-3" style={{marginBottom:16}}>
        {[
          { label:'Critical Alerts', value:criticalCount, color:'var(--danger)',  bg:'rgba(255,56,96,.08)',  border:'rgba(255,56,96,.2)' },
          { label:'Warnings',        value:warningCount,  color:'var(--warn)',    bg:'rgba(255,190,11,.08)', border:'rgba(255,190,11,.2)' },
          { label:'Models Monitored',value:Object.keys(drift).length, color:'var(--accent)', bg:'rgba(0,212,255,.08)', border:'rgba(0,212,255,.2)' },
        ].map((s,i) => (
          <div key={i} className="card" style={{border:`1px solid ${s.border}`,background:s.bg}}>
            <div className="card-label">{s.label}</div>
            <div className="card-value" style={{color:s.color,fontSize:36}}>{s.value}</div>
          </div>
        ))}
      </div>

      {/* Drift + radar */}
      <div className="grid-2">
        <div className="card">
          <div className="card-head">
            <span className="card-title">Concept Drift (PSI)</span>
            <button className="btn btn-secondary btn-sm" onClick={load}>↺</button>
          </div>
          {loading
            ? <div className="empty"><span className="spinner"/></div>
            : Object.keys(drift).length === 0
              ? <div className="empty"><div className="empty-icon">📊</div>No drift data yet — predictions needed to build baseline.</div>
              : Object.entries(drift).map(([k,v]) =>
                  v?.psi !== undefined
                    ? <DriftGauge key={k} name={k.toUpperCase()} psi={v.psi} status={v.drift_status}/>
                    : <div key={k} style={{fontSize:10,color:'var(--text3)',marginBottom:6}}>{k.toUpperCase()}: {v?.status || 'no data'}</div>
                )
          }
          <div style={{display:'flex',gap:12,marginTop:10,fontSize:9}}>
            <span style={{color:'#00ff9d'}}>■ OK (&lt;0.05)</span>
            <span style={{color:'#ffbe0b'}}>■ WARNING (0.05–0.20)</span>
            <span style={{color:'#ff3860'}}>■ DRIFT (&gt;0.20)</span>
          </div>
        </div>

        <div className="card">
          <div className="card-head">
            <span className="card-title">DSO Drift Radar</span>
          </div>
          {radarData.length === 0
            ? <div className="empty"><div className="empty-icon">📡</div>Awaiting drift data</div>
            : <ResponsiveContainer width="100%" height={220}>
                <RadarChart data={radarData}>
                  <PolarGrid stroke="#1a2535"/>
                  <PolarAngleAxis dataKey="model" tick={{fill:'#7a8fa6',fontSize:10,fontFamily:'Space Mono'}}/>
                  <Radar dataKey="psi" stroke="#00d4ff" fill="#00d4ff" fillOpacity={.2} strokeWidth={1.5}/>
                  <Tooltip contentStyle={{background:'#101820',border:'1px solid #243044',fontSize:10}}/>
                </RadarChart>
              </ResponsiveContainer>
          }
        </div>
      </div>

      {/* Alerts */}
      <div className="card">
        <div className="card-head">
          <span className="card-title">Active Alerts</span>
          <div style={{display:'flex',gap:6}}>
            {['','critical','warning','info'].map(s => (
              <button key={s} className={`btn btn-sm ${filter===s?'btn-primary':'btn-secondary'}`}
                onClick={()=>setFilter(s)} style={{fontSize:9,padding:'4px 10px'}}>
                {s || 'All'}
              </button>
            ))}
          </div>
        </div>

        {filteredAlerts.length === 0
          ? <div className="empty"><div className="empty-icon">✅</div>{filter ? `No ${filter} alerts.` : 'All systems nominal — no alerts.'}</div>
          : filteredAlerts.map(a => (
              <div key={a.alert_id} className={`alert-item ${a.severity}`}>
                <span style={{fontSize:14,marginTop:1}}>{SEVERITY_ICON[a.severity] || '⚪'}</span>
                <div className="alert-body">
                  <div className="alert-msg">{a.message}</div>
                  <div className="alert-meta">
                    {a.model_name?.toUpperCase()} · {a.metric?.toUpperCase()}: {a.value?.toFixed(4)} (threshold {a.threshold?.toFixed(4)}) · {new Date(a.created_at).toLocaleTimeString()}
                  </div>
                </div>
              </div>
            ))
        }
      </div>
    </div>
  )
}
