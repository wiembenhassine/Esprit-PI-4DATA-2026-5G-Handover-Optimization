import { useEffect, useState } from 'react'
import { getRegistry, triggerTraining, getJobs, deployModel } from '../api'
import { useAuth } from '../AuthContext'

const DSO_COLOR  = { dso1:'#00d4ff', dso2:'#00ff9d', dso3:'#ff6b35', dso4:'#ffbe0b' }
const DSO_DESC   = {
  dso1: 'Signal Degradation · XGBClassifier + Keras NN · 16 features',
  dso2: 'Neighbor Gain Estimation · XGBRegressor(honest) + Keras NN · 7 features',
  dso3: 'User State Profiling · KMeans (k=4) · 7 features',
  dso4: 'Master HO Controller · XGBClassifier · 18 features (incl. DSO1-3 outputs)',
}

function StatusBadge({ status }) {
  const map = {
    deployed: { color:'#00ff9d', bg:'rgba(0,255,157,.1)', border:'rgba(0,255,157,.3)', label:'● DEPLOYED' },
    trained:  { color:'#ffbe0b', bg:'rgba(255,190,11,.1)',border:'rgba(255,190,11,.3)',label:'● TRAINED' },
    retired:  { color:'#3d5166', bg:'rgba(61,81,102,.1)', border:'rgba(61,81,102,.3)', label:'● RETIRED' },
    training: { color:'#00d4ff', bg:'rgba(0,212,255,.1)', border:'rgba(0,212,255,.3)', label:'⟳ TRAINING' },
  }
  const s = map[status] || map.retired
  return (
    <span style={{ fontSize:9, color:s.color, background:s.bg, border:`1px solid ${s.border}`, padding:'2px 8px', borderRadius:4, fontFamily:'var(--font-head)', fontWeight:600 }}>
      {s.label}
    </span>
  )
}

export default function MLOps() {
  const { isAdmin } = useAuth()
  const [registry, setRegistry] = useState({})
  const [jobs,     setJobs]     = useState([])
  const [training, setTraining] = useState(null)   // which model is training
  const [deploying,setDeploying]= useState(null)
  const [toast,    setToast]    = useState('')
  const [loading,  setLoading]  = useState(true)

  function showToast(msg) {
    setToast(msg)
    setTimeout(() => setToast(''), 3500)
  }

  async function load() {
    setLoading(true)
    try {
      const [reg, j] = await Promise.all([getRegistry(), getJobs()])
      setRegistry(reg || {})
      setJobs((j?.jobs || []).slice(-10).reverse())
    } catch(e) {
      console.error(e)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  async function handleTrain(model_name) {
    setTraining(model_name)
    try {
      await triggerTraining(model_name)
      showToast(`Training job started for ${model_name.toUpperCase()}`)
      setTimeout(load, 3000)   // refresh after ~3s
    } catch(e) {
      showToast(`Error: ${e.message}`)
    } finally {
      setTraining(null)
    }
  }

  async function handleDeploy(model_name, version) {
    setDeploying(`${model_name}-${version}`)
    try {
      await deployModel(model_name, version)
      showToast(`${model_name.toUpperCase()} ${version} deployed ✓`)
      load()
    } catch(e) {
      showToast(`Deploy failed: ${e.message}`)
    } finally {
      setDeploying(null)
    }
  }

  return (
    <div className="fade-up">
      {/* Toast */}
      {toast && (
        <div className="alert-item info fade-in" style={{marginBottom:14}}>
          <span>ℹ</span>
          <div className="alert-body"><div className="alert-msg">{toast}</div></div>
        </div>
      )}

      {/* Model Registry */}
      <div style={{display:'flex',alignItems:'center',justifyContent:'space-between',marginBottom:14}}>
        <div style={{fontFamily:'var(--font-head)',fontWeight:700,fontSize:13}}>Model Registry</div>
        <button className="btn btn-secondary btn-sm" onClick={load}>↺ Refresh</button>
      </div>

      {loading
        ? <div className="empty"><span className="spinner" style={{width:20,height:20,borderWidth:2}}/></div>
        : <div className="grid-2">
            {Object.entries(registry).map(([name, versions]) => {
              const color = DSO_COLOR[name] || '#7a8fa6'
              const deployed = versions?.find(v => v.status === 'deployed')
              const allVersions = [...(versions || [])].reverse()
              return (
                <div key={name} className="model-card" style={{borderLeft:`3px solid ${color}`}}>
                  <div style={{display:'flex',alignItems:'flex-start',justifyContent:'space-between',marginBottom:10}}>
                    <div>
                      <div style={{fontFamily:'var(--font-head)',fontWeight:800,fontSize:16,color}}>{name.toUpperCase()}</div>
                      <div style={{fontSize:9,color:'var(--text3)',marginTop:2,maxWidth:220}}>{DSO_DESC[name]}</div>
                    </div>
                    {deployed && <StatusBadge status="deployed"/>}
                  </div>

                  {/* Active version metrics */}
                  {deployed && (
                    <div style={{marginBottom:10}}>
                      <div style={{fontSize:9,color:'var(--text3)',letterSpacing:'1px',textTransform:'uppercase',marginBottom:6}}>
                        Active: {deployed.version}
                      </div>
                      <div className="model-metrics">
                        {Object.entries(deployed.metrics || {}).map(([k,v]) => (
                          <div key={k} className="metric-row">
                            <span className="metric-key">{k.replace(/_/g,' ').toUpperCase()}</span>
                            <span className="metric-val" style={{color}}>{typeof v==='number'?v.toFixed(3):v}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* All versions */}
                  {allVersions.length > 1 && (
                    <div style={{marginBottom:10}}>
                      <div style={{fontSize:9,color:'var(--text3)',letterSpacing:'1px',textTransform:'uppercase',marginBottom:6}}>All Versions</div>
                      {allVersions.map(v => (
                        <div key={v.version} style={{display:'flex',alignItems:'center',gap:8,marginBottom:4}}>
                          <span style={{fontSize:10,color:'var(--text2)',fontFamily:'var(--font-mono)'}}>{v.version}</span>
                          <StatusBadge status={v.status}/>
                          {isAdmin && v.status === 'trained' && (
                            <button
                              className="btn btn-sm btn-secondary"
                              style={{fontSize:8,padding:'2px 8px',marginLeft:'auto'}}
                              disabled={deploying === `${name}-${v.version}`}
                              onClick={() => handleDeploy(name, v.version)}
                            >
                              {deploying === `${name}-${v.version}` ? <span className="spinner" style={{width:8,height:8,borderWidth:1}}/> : 'Deploy'}
                            </button>
                          )}
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Retrain button */}
                  <button
                    className="btn btn-secondary btn-sm btn-full"
                    onClick={() => handleTrain(name)}
                    disabled={training === name}
                    style={{marginTop:4}}
                  >
                    {training === name
                      ? <><span className="spinner" style={{width:10,height:10,borderWidth:1.5}}/>Training…</>
                      : `↺ Retrain ${name.toUpperCase()}`
                    }
                  </button>
                </div>
              )
            })}
          </div>
      }

      {/* Training Jobs */}
      <div className="card" style={{marginTop:16}}>
        <div className="card-head">
          <span className="card-title">Training Jobs</span>
          <span className="chip">{jobs.length} recent</span>
        </div>
        {jobs.length === 0
          ? <div className="empty"><div className="empty-icon">⚙</div>No training jobs yet.</div>
          : <div className="table-wrap">
              <table>
                <thead>
                  <tr><th>Job ID</th><th>Model</th><th>Status</th><th>Version</th><th>Metrics</th><th>Started</th></tr>
                </thead>
                <tbody>
                  {jobs.map(j => (
                    <tr key={j.job_id}>
                      <td style={{fontFamily:'var(--font-mono)',fontSize:10,color:'var(--text3)'}}>{j.job_id?.slice(-10)}</td>
                      <td><span style={{color:DSO_COLOR[j.model_name],fontFamily:'var(--font-head)',fontWeight:700,fontSize:11}}>{j.model_name?.toUpperCase()}</span></td>
                      <td><StatusBadge status={j.status}/></td>
                      <td style={{fontSize:10,color:'var(--text2)'}}>{j.new_version || '—'}</td>
                      <td style={{fontSize:9,color:'var(--text3)'}}>
                        {j.metrics ? Object.entries(j.metrics).map(([k,v])=>`${k}: ${v}`).join(' · ') : '—'}
                      </td>
                      <td style={{fontSize:10,color:'var(--text3)'}}>{j.started_at ? new Date(j.started_at).toLocaleTimeString() : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
        }
      </div>
    </div>
  )
}
