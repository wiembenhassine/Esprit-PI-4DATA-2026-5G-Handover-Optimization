import { useState } from 'react'
import { predict } from '../api'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'

const DEFAULTS = {
  scenario:'hbahn', rsrp:'-95', rsrq:'-13', sinr:'5',
  cqi:'9', tx_power:'20', ta:'3', velocity:'75',
  n_neighbors:'2', neighbor_gap:'3.0', best_neighbor_rsrp:'-92',
  rsrp_slope3:'-1.5', sinr_slope3:'-0.5',
  hour_of_day:'8', day_of_week:'1',
  cell_hist_datarate_mean:'15', cell_load_drop_flag:'0',
  latency_is_imputed:'1',
}

function ResultDSO({ title, children, color }) {
  return (
    <div className="result-dso" style={{ borderLeft:`2px solid ${color}` }}>
      <div className="result-dso-head">{title}</div>
      {children}
    </div>
  )
}

function ProbBar({ value, color }) {
  return (
    <div className="prob-bar" style={{marginTop:6}}>
      <div className="prob-track">
        <div className="prob-fill" style={{ width:`${value*100}%`, background:color }} />
      </div>
      <span className="prob-label">{(value*100).toFixed(1)}%</span>
    </div>
  )
}

export default function Predict() {
  const [form,    setForm]    = useState(DEFAULTS)
  const [result,  setResult]  = useState(null)
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState('')

  const set = (k, v) => setForm(f => ({...f, [k]:v}))

  async function handleSubmit(e) {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      const payload = {
        timestamp: Date.now() / 1000,
        scenario:  form.scenario,
        rsrp:      parseFloat(form.rsrp),
        rsrq:      parseFloat(form.rsrq),
        sinr:      parseFloat(form.sinr),
        cqi:       parseInt(form.cqi),
        tx_power:  parseFloat(form.tx_power),
        ta:        parseFloat(form.ta),
        velocity:  parseFloat(form.velocity),
        n_neighbors: parseInt(form.n_neighbors),
        neighbor_gap: parseFloat(form.neighbor_gap),
        best_neighbor_rsrp: parseFloat(form.best_neighbor_rsrp),
        rsrp_slope3: parseFloat(form.rsrp_slope3),
        sinr_slope3: parseFloat(form.sinr_slope3),
        hour_of_day: parseInt(form.hour_of_day),
        day_of_week: parseInt(form.day_of_week),
        cell_hist_datarate_mean: parseFloat(form.cell_hist_datarate_mean),
        cell_load_drop_flag: parseInt(form.cell_load_drop_flag),
        latency_is_imputed: parseInt(form.latency_is_imputed),
      }
      const res = await predict(payload)
      setResult(res)
    } catch(e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  function handleReset() { setForm(DEFAULTS); setResult(null); setError('') }

  const shapData = result?.dso1?.shap_values
    ? Object.entries(result.dso1.shap_values)
        .sort((a,b) => Math.abs(b[1]) - Math.abs(a[1]))
        .map(([k,v]) => ({ feature:k.replace(/_/g,' '), value:v }))
    : []

  const d1 = result?.dso1; const d2 = result?.dso2
  const d3 = result?.dso3; const d4 = result?.dso4

  return (
    <div className="fade-up">
      <div className="grid-2">
        {/* ── Form ── */}
        <div className="card">
          <div className="card-head">
            <span className="card-title">Input Features</span>
            <span className="chip chip-blue">DSO Chain</span>
          </div>
          <form onSubmit={handleSubmit}>
            {/* Scenario */}
            <div style={{marginBottom:10}}>
              <div className="form-group">
                <label className="form-label">Scenario</label>
                <select className="form-select" value={form.scenario} onChange={e=>set('scenario',e.target.value)}>
                  <option value="hbahn">H-Bahn (Light Rail)</option>
                  <option value="mobile">Mobile (Pedestrian)</option>
                  <option value="static">Static (Fixed)</option>
                </select>
              </div>
            </div>

            {/* RF Metrics */}
            <div style={{fontSize:9,color:'var(--text3)',letterSpacing:'1.5px',textTransform:'uppercase',marginBottom:8}}>RF Metrics</div>
            <div className="form-grid-3" style={{marginBottom:10}}>
              {[['rsrp','RSRP (dBm)'],['rsrq','RSRQ (dB)'],['sinr','SINR (dB)'],
                ['cqi','CQI'],['tx_power','TX Power (dBm)'],['ta','TA']].map(([k,l])=>(
                <div key={k} className="form-group">
                  <label className="form-label">{l}</label>
                  <input className="form-input" type="number" step="0.1" value={form[k]} onChange={e=>set(k,e.target.value)}/>
                </div>
              ))}
            </div>

            {/* Mobility */}
            <div style={{fontSize:9,color:'var(--text3)',letterSpacing:'1.5px',textTransform:'uppercase',marginBottom:8}}>Mobility & Neighbors</div>
            <div className="form-grid-3" style={{marginBottom:10}}>
              {[['velocity','Velocity (km/h)'],['n_neighbors','Num Neighbors'],['neighbor_gap','Neighbor Gap (dB)'],
                ['best_neighbor_rsrp','Best Neighbor RSRP']].map(([k,l])=>(
                <div key={k} className="form-group">
                  <label className="form-label">{l}</label>
                  <input className="form-input" type="number" step="0.1" value={form[k]} onChange={e=>set(k,e.target.value)}/>
                </div>
              ))}
            </div>

            {/* Trend */}
            <div style={{fontSize:9,color:'var(--text3)',letterSpacing:'1.5px',textTransform:'uppercase',marginBottom:8}}>Signal Trend</div>
            <div className="form-grid-3" style={{marginBottom:10}}>
              {[['rsrp_slope3','RSRP Slope-3'],['sinr_slope3','SINR Slope-3']].map(([k,l])=>(
                <div key={k} className="form-group">
                  <label className="form-label">{l}</label>
                  <input className="form-input" type="number" step="0.1" value={form[k]} onChange={e=>set(k,e.target.value)}/>
                </div>
              ))}
            </div>

            {/* Context */}
            <div style={{fontSize:9,color:'var(--text3)',letterSpacing:'1.5px',textTransform:'uppercase',marginBottom:8}}>Context</div>
            <div className="form-grid-3" style={{marginBottom:16}}>
              {[['hour_of_day','Hour of Day'],['day_of_week','Day of Week'],
                ['cell_hist_datarate_mean','Cell Hist. Datarate'],
                ['cell_load_drop_flag','Load Drop Flag (0/1)'],
                ['latency_is_imputed','Latency Imputed (0/1)']].map(([k,l])=>(
                <div key={k} className="form-group">
                  <label className="form-label">{l}</label>
                  <input className="form-input" type="number" step="1" value={form[k]} onChange={e=>set(k,e.target.value)}/>
                </div>
              ))}
            </div>

            {error && <div className="alert-item warning" style={{marginBottom:10}}><span>⚠</span><div className="alert-body"><div className="alert-msg">{error}</div></div></div>}

            <div style={{display:'flex',gap:8}}>
              <button className="btn btn-primary" type="submit" disabled={loading} style={{flex:1}}>
                {loading ? <><span className="spinner"/>Running Pipeline…</> : '▶  Run DSO Pipeline'}
              </button>
              <button className="btn btn-secondary btn-sm" type="button" onClick={handleReset}>Reset</button>
            </div>
          </form>
        </div>

        {/* ── Results ── */}
        <div className="card">
          <div className="card-head">
            <span className="card-title">Pipeline Results</span>
            {result && <span className="chip">{result.latency_ms}ms</span>}
          </div>

          {!result && !loading && (
            <div className="empty" style={{marginTop:40}}>
              <div className="empty-icon">🤖</div>
              Fill the form and click Run to see DSO predictions here.
            </div>
          )}
          {loading && (
            <div className="empty" style={{marginTop:40}}>
              <div style={{marginBottom:12}}><span className="spinner" style={{width:24,height:24,borderWidth:3}}/></div>
              Running DSO1→DSO2→DSO3→DSO4 chain…
            </div>
          )}

          {result && !loading && (
            <>
              <div className="result-grid">
                {/* DSO1 */}
                <ResultDSO title="DSO1 — Signal Risk" color="#00d4ff">
                  <div className="result-dso-value" style={{color: d1.is_degrading ? '#ff3860' : '#00ff9d'}}>
                    {d1.is_degrading ? '⚠ DEGRADING' : '✓ STABLE'}
                  </div>
                  <ProbBar value={d1.degradation_prob} color={d1.degradation_prob>.5?'#ff3860':'#00ff9d'}/>
                  <div style={{fontSize:9,color:'var(--text3)',marginTop:6}}>RSRP+5 → {d1.rsrp_future_5_pred} dBm</div>
                </ResultDSO>

                {/* DSO2 */}
                <ResultDSO title="DSO2 — HO Target" color="#00ff9d">
                  <div className="result-dso-value" style={{color:'#00d4ff'}}>
                    +{d2.predicted_neighbor_gap?.toFixed(2)} dB
                  </div>
                  <div style={{fontSize:10,color:'var(--text2)'}}>Predicted gain</div>
                  <div style={{fontSize:9,color:'var(--text3)',marginTop:4}}>Best RSRP: {d2.predicted_best_rsrp} dBm</div>
                </ResultDSO>

                {/* DSO3 */}
                <ResultDSO title="DSO3 — Network State" color="#ff6b35">
                  <div className="result-dso-value">
                    <span className={`tag tag-${d3.cluster_label}`} style={{fontSize:14,padding:'4px 10px'}}>
                      {d3.cluster_label}
                    </span>
                  </div>
                  <div style={{fontSize:9,color:'var(--text3)',marginTop:10}}>
                    Cluster {d3.cluster_id} — confidence {d3.cluster_probs?.[d3.cluster_id]
                      ? (d3.cluster_probs[d3.cluster_id]*100).toFixed(0)+'%' : '—'}
                  </div>
                </ResultDSO>

                {/* DSO4 */}
                <ResultDSO title="DSO4 — Decision" color="#ffbe0b">
                  <div className="result-dso-value" style={{color: d4.handover_recommended ? '#ff3860' : '#00ff9d'}}>
                    {d4.handover_recommended ? '⚡ HANDOVER' : '✓ HOLD'}
                  </div>
                  <ProbBar value={d4.handover_prob} color={d4.handover_recommended?'#ff3860':'#00ff9d'}/>
                  <div style={{fontSize:9,color:'var(--text3)',marginTop:4}}>
                    Confidence: <span style={{color: d4.confidence==='high'?'#00ff9d':d4.confidence==='medium'?'#ffbe0b':'#7a8fa6'}}>
                      {d4.confidence?.toUpperCase()}
                    </span>
                  </div>
                </ResultDSO>
              </div>

              {/* SHAP */}
              {shapData.length > 0 && (
                <div style={{marginTop:16}}>
                  <div className="card-label" style={{marginBottom:10}}>DSO1 SHAP Feature Importance</div>
                  <ResponsiveContainer width="100%" height={130}>
                    <BarChart data={shapData} layout="vertical" margin={{left:10,right:20,top:0,bottom:0}}>
                      <XAxis type="number" tick={{fill:'#3d5166',fontSize:9}}/>
                      <YAxis type="category" dataKey="feature" tick={{fill:'#7a8fa6',fontSize:9,fontFamily:'Space Mono'}} width={110}/>
                      <Tooltip contentStyle={{background:'#101820',border:'1px solid #243044',fontSize:10}}/>
                      <Bar dataKey="value" radius={[0,3,3,0]}>
                        {shapData.map((e,i)=><Cell key={i} fill={e.value>0?'#ff3860':'#00ff9d'}/>)}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                  <div style={{display:'flex',gap:16,marginTop:4}}>
                    <span style={{fontSize:9,color:'#ff3860'}}>■ Increases risk</span>
                    <span style={{fontSize:9,color:'#00ff9d'}}>■ Reduces risk</span>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}
