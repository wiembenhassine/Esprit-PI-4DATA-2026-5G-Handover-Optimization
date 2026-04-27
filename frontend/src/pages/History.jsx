import { useEffect, useState } from 'react'
import { getPredictHistory } from '../api'

export default function History() {
  const [items,    setItems]    = useState([])
  const [loading,  setLoading]  = useState(true)
  const [scenario, setScenario] = useState('')
  const [search,   setSearch]   = useState('')
  const [total,    setTotal]    = useState(0)
  const limit = 100

  async function load(sc) {
    setLoading(true)
    try {
      const d = await getPredictHistory(limit, sc || undefined)
      setItems(d?.items || [])
      setTotal(d?.total || 0)
    } catch(e) {
      setItems([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load(scenario) }, [scenario])

  const filtered = items.filter(p => {
    if (!search) return true
    const s = search.toLowerCase()
    return (
      p.request_id?.includes(s) ||
      p.scenario?.includes(s)   ||
      p.dso3?.cluster_label?.includes(s)
    )
  })

  return (
    <div className="fade-up">
      {/* Controls */}
      <div className="card" style={{marginBottom:14}}>
        <div style={{display:'flex',gap:10,alignItems:'flex-end'}}>
          <div className="form-group" style={{flex:'0 0 160px'}}>
            <label className="form-label">Scenario Filter</label>
            <select className="form-select" value={scenario} onChange={e=>setScenario(e.target.value)}>
              <option value="">All Scenarios</option>
              <option value="hbahn">H-Bahn</option>
              <option value="mobile">Mobile</option>
              <option value="static">Static</option>
            </select>
          </div>
          <div className="form-group" style={{flex:1}}>
            <label className="form-label">Search (ID / scenario / state)</label>
            <input className="form-input" placeholder="Search…" value={search} onChange={e=>setSearch(e.target.value)}/>
          </div>
          <button className="btn btn-secondary btn-sm" onClick={()=>load(scenario)}>↺ Refresh</button>
        </div>
      </div>

      {/* Summary chips */}
      <div style={{display:'flex',gap:8,marginBottom:14,flexWrap:'wrap'}}>
        <span className="chip">Total: {total}</span>
        <span className="chip chip-blue">Showing: {filtered.length}</span>
        {filtered.length > 0 && (
          <>
            <span className="chip chip-green">
              HO rate: {(filtered.filter(p=>p.dso4?.handover_recommended).length/filtered.length*100).toFixed(0)}%
            </span>
            <span className="chip chip-orange">
              Deg rate: {(filtered.filter(p=>p.dso1?.is_degrading).length/filtered.length*100).toFixed(0)}%
            </span>
          </>
        )}
      </div>

      <div className="card">
        <div className="card-head">
          <span className="card-title">Prediction History</span>
        </div>

        {loading && (
          <div className="empty"><span className="spinner" style={{width:20,height:20,borderWidth:2}}/></div>
        )}

        {!loading && filtered.length === 0 && (
          <div className="empty">
            <div className="empty-icon">📋</div>
            No predictions found. Run some via the Predict page first.
          </div>
        )}

        {!loading && filtered.length > 0 && (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Request ID</th>
                  <th>Time</th>
                  <th>Scenario</th>
                  <th>DSO1 Risk</th>
                  <th>DSO2 Gain</th>
                  <th>DSO3 State</th>
                  <th>DSO4 Decision</th>
                  <th>Conf.</th>
                  <th>Latency</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map(p => {
                  const d1=p.dso1; const d2=p.dso2; const d3=p.dso3; const d4=p.dso4
                  const ts = new Date(p.created_at || p.timestamp*1000).toLocaleTimeString()
                  return (
                    <tr key={p.request_id}>
                      <td style={{fontFamily:'var(--font-mono)',fontSize:10,color:'var(--text3)'}}>{p.request_id?.slice(-10)}</td>
                      <td style={{fontSize:10,color:'var(--text3)'}}>{ts}</td>
                      <td><span className={`tag tag-${p.scenario}`}>{p.scenario}</span></td>
                      <td>
                        <div style={{display:'flex',alignItems:'center',gap:6}}>
                          <div style={{width:50,height:4,background:'var(--bg3)',borderRadius:2,overflow:'hidden'}}>
                            <div style={{width:`${(d1?.degradation_prob||0)*100}%`,height:'100%',background:d1?.is_degrading?'#ff3860':'#00ff9d',borderRadius:2}}/>
                          </div>
                          <span style={{fontSize:10,color:'var(--text2)'}}>{((d1?.degradation_prob||0)*100).toFixed(0)}%</span>
                        </div>
                      </td>
                      <td style={{fontSize:10,color:'#00d4ff'}}>+{d2?.predicted_neighbor_gap?.toFixed(1)} dB</td>
                      <td><span className={`tag tag-${d3?.cluster_label}`}>{d3?.cluster_label}</span></td>
                      <td>
                        <span className={d4?.handover_recommended ? 'badge-ho' : 'badge-ok'}>
                          {d4?.handover_recommended ? '⚡ HO' : '✓ HOLD'}
                        </span>
                      </td>
                      <td style={{fontSize:10,color:d4?.confidence==='high'?'#00ff9d':d4?.confidence==='medium'?'#ffbe0b':'#7a8fa6'}}>
                        {d4?.confidence?.toUpperCase()}
                      </td>
                      <td style={{fontSize:10,color:'var(--text3)'}}>{p.latency_ms}ms</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
