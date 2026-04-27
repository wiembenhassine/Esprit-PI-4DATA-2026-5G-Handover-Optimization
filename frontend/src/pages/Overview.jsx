import { useEffect, useState, useRef } from 'react'
import { getKPIs, getLiveFeed, getPredictStats } from '../api'
import {
  AreaChart, Area, BarChart, Bar, XAxis, YAxis,
  CartesianGrid, Tooltip, ResponsiveContainer, Cell,
} from 'recharts'

function ProbBar({ value, color }) {
  return (
    <div className="prob-bar">
      <div className="prob-track">
        <div className="prob-fill" style={{ width:`${value*100}%`, background:color }} />
      </div>
      <span className="prob-label">{(value*100).toFixed(0)}%</span>
    </div>
  )
}

function KPICard({ label, value, sub, accent }) {
  return (
    <div className={`card card-top-${accent} fade-up`}>
      <div className="card-label">{label}</div>
      <div className="card-value">{value ?? '—'}</div>
      <div className="card-sub">{sub}</div>
    </div>
  )
}

function genPoint(t, prev) {
  const rsrp  = prev ? prev.rsrp  + (Math.random()-.5)*3  : -88
  const risk  = prev ? Math.max(0,Math.min(1, prev.risk  + (Math.random()-.5)*.08)) : .3
  return { t, rsrp: Math.max(-120, Math.min(-60, rsrp)), risk }
}

const CLUSTER_COLORS = { good:'#00ff9d', fair:'#00d4ff', cell_edge:'#ff6b35', congested:'#ff3860' }

export default function Overview() {
  const [kpis,  setKpis]  = useState(null)
  const [feed,  setFeed]  = useState([])
  const [chart, setChart] = useState(() => Array.from({length:30},(_,i)=>genPoint(i,null)))
  const [error, setError] = useState('')
  const tick = useRef(30)

  useEffect(() => {
    getKPIs().then(setKpis).catch(e => setError(e.message))
    getLiveFeed(15).then(d => setFeed(d?.items || [])).catch(() => {})

    const iv = setInterval(() => {
      setChart(prev => {
        const last = prev[prev.length-1]
        return [...prev.slice(1), genPoint(tick.current++, last)]
      })
      // Refresh feed every 10s
      if (tick.current % 5 === 0) {
        getLiveFeed(15).then(d => setFeed(d?.items || [])).catch(()=>{})
      }
    }, 2000)
    return () => clearInterval(iv)
  }, [])

  const clusterData = kpis?.network_state_distribution
    ? Object.entries(kpis.network_state_distribution).map(([k,v])=>({ name:k, value:v, color:CLUSTER_COLORS[k] || '#7a8fa6' }))
    : []

  return (
    <div className="fade-up">
      {error && <div className="alert-item warning" style={{marginBottom:14}}><span>⚠</span><div className="alert-body"><div className="alert-msg">{error}</div></div></div>}

      {/* KPIs */}
      <div className="grid-4">
        <KPICard label="Total Predictions" value={kpis ? (kpis.total_predictions/1000).toFixed(1)+'K' : null} sub="all time" accent="blue" />
        <KPICard label="Handover Rate"     value={kpis ? kpis.handover_rate_pct+'%' : null}  sub="DSO4 decisions"   accent="green" />
        <KPICard label="Degradation Rate"  value={kpis ? kpis.degradation_rate_pct+'%' : null} sub="DSO1 risk flag"  accent="orange" />
        <KPICard label="Avg Latency"       value={kpis ? kpis.avg_inference_latency_ms+'ms' : null} sub={`p95 ${kpis?.p95_inference_latency_ms||'—'}ms`} accent="blue" />
      </div>

      {/* Charts */}
      <div className="grid-2-1">
        <div className="card fade-up">
          <div className="card-head">
            <span className="card-title">RSRP Signal + Degradation Risk</span>
            <span className="chip chip-blue">LIVE</span>
          </div>
          <div style={{position:'relative'}}>
            <div className="scan" />
            <ResponsiveContainer width="100%" height={190}>
              <AreaChart data={chart} margin={{top:4,right:4,left:-20,bottom:0}}>
                <defs>
                  <linearGradient id="gRsrp" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%"  stopColor="#00d4ff" stopOpacity={.3}/>
                    <stop offset="95%" stopColor="#00d4ff" stopOpacity={0}/>
                  </linearGradient>
                  <linearGradient id="gRisk" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%"  stopColor="#ff3860" stopOpacity={.25}/>
                    <stop offset="95%" stopColor="#ff3860" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#1a2535"/>
                <XAxis dataKey="t" hide/>
                <YAxis yAxisId="l" domain={[-120,-60]} tick={{fill:'#3d5166',fontSize:9}}/>
                <YAxis yAxisId="r" orientation="right" domain={[0,1]} tick={{fill:'#3d5166',fontSize:9}}/>
                <Tooltip contentStyle={{background:'#101820',border:'1px solid #243044',fontSize:10,fontFamily:'Space Mono'}}/>
                <Area yAxisId="l" type="monotone" dataKey="rsrp" stroke="#00d4ff" strokeWidth={1.5} fill="url(#gRsrp)" dot={false} name="RSRP (dBm)"/>
                <Area yAxisId="r" type="monotone" dataKey="risk" stroke="#ff3860" strokeWidth={1}   fill="url(#gRisk)" dot={false} name="Risk Score"/>
              </AreaChart>
            </ResponsiveContainer>
          </div>
          <div style={{display:'flex',gap:16,marginTop:6}}>
            <span style={{fontSize:9,color:'#00d4ff'}}>■ RSRP (dBm)</span>
            <span style={{fontSize:9,color:'#ff3860'}}>■ Degradation Risk</span>
          </div>
        </div>

        <div className="card fade-up">
          <div className="card-head">
            <span className="card-title">Network State</span>
            <span className="chip chip-green">DSO3</span>
          </div>
          {clusterData.length === 0
            ? <div className="empty"><div className="empty-icon">📊</div>No data yet</div>
            : <ResponsiveContainer width="100%" height={190}>
                <BarChart data={clusterData} layout="vertical" margin={{top:4,right:20,left:0,bottom:4}}>
                  <XAxis type="number" tick={{fill:'#3d5166',fontSize:9}}/>
                  <YAxis type="category" dataKey="name" tick={{fill:'#7a8fa6',fontSize:10,fontFamily:'Space Mono'}} width={70}/>
                  <Tooltip contentStyle={{background:'#101820',border:'1px solid #243044',fontSize:10}}/>
                  <Bar dataKey="value" radius={[0,3,3,0]}>
                    {clusterData.map((e,i)=><Cell key={i} fill={e.color}/>)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
          }
        </div>
      </div>

      {/* Live feed table */}
      <div className="card fade-up">
        <div className="card-head">
          <span className="card-title">Live Prediction Feed</span>
          <div style={{display:'flex',gap:8,alignItems:'center'}}>
            <div style={{display:'flex',alignItems:'center',gap:5,fontSize:10,color:'#00ff9d'}}>
              <span className="live-dot"/>LIVE
            </div>
            <span className="chip">{feed.length} rows</span>
          </div>
        </div>
        {feed.length === 0
          ? <div className="empty"><div className="empty-icon">📡</div>No predictions yet — run some via the Predict page.</div>
          : <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>ID</th><th>Scenario</th><th>State</th>
                    <th>DSO1 Risk</th><th>DSO4 Decision</th><th>Conf.</th><th>Latency</th>
                  </tr>
                </thead>
                <tbody>
                  {feed.map(row => {
                    const riskColor = row.risk_score > .6 ? '#ff3860' : row.risk_score > .35 ? '#ffbe0b' : '#00ff9d'
                    return (
                      <tr key={row.request_id}>
                        <td style={{fontFamily:'var(--font-mono)',fontSize:10,color:'var(--text3)'}}>{row.request_id?.slice(-8)}</td>
                        <td><span className={`tag tag-${row.scenario}`}>{row.scenario}</span></td>
                        <td><span className={`tag tag-${row.network_state}`}>{row.network_state}</span></td>
                        <td style={{minWidth:120}}><ProbBar value={row.risk_score||0} color={riskColor}/></td>
                        <td><span className={row.handover ? 'badge-ho' : 'badge-ok'}>{row.handover ? '⚡ HANDOVER' : '✓ HOLD'}</span></td>
                        <td style={{fontSize:10,color:row.confidence==='high'?'#00ff9d':row.confidence==='medium'?'#ffbe0b':'#7a8fa6'}}>{row.confidence?.toUpperCase()}</td>
                        <td style={{fontSize:10,color:'var(--text3)'}}>{row.latency_ms}ms</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
        }
      </div>
    </div>
  )
}
