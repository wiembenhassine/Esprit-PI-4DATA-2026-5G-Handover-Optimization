import { useState } from 'react'
import { useAuth } from '../AuthContext'
import { useNavigate } from 'react-router-dom'

const CSS = `
.login-root {
  min-height: 100vh;
  background: #060a0f;
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
  overflow: hidden;
}

/* Animated grid background */
.login-grid {
  position: absolute;
  inset: 0;
  background-image:
    linear-gradient(rgba(0,212,255,.04) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,212,255,.04) 1px, transparent 1px);
  background-size: 40px 40px;
  mask-image: radial-gradient(ellipse 80% 80% at 50% 50%, black 40%, transparent 100%);
}

/* Glowing orbs */
.login-orb {
  position: absolute;
  border-radius: 50%;
  filter: blur(80px);
  opacity: .18;
  animation: orbFloat 8s ease-in-out infinite;
}
.login-orb-1 { width:400px; height:400px; background:#00d4ff; top:-100px; left:-100px; animation-delay:0s; }
.login-orb-2 { width:300px; height:300px; background:#00ff9d; bottom:-80px;  right:-80px;  animation-delay:3s; }
.login-orb-3 { width:200px; height:200px; background:#ff6b35; top:50%; left:60%; animation-delay:5s; }
@keyframes orbFloat { 0%,100%{transform:translateY(0) scale(1)} 50%{transform:translateY(-20px) scale(1.05)} }

/* Card */
.login-card {
  position: relative;
  z-index: 10;
  width: 100%;
  max-width: 420px;
  background: rgba(11,16,24,.92);
  border: 1px solid #1a2535;
  border-radius: 16px;
  padding: 40px;
  backdrop-filter: blur(20px);
  box-shadow: 0 0 80px rgba(0,0,0,.5), 0 0 0 1px rgba(0,212,255,.06);
  animation: cardIn .5s cubic-bezier(.16,1,.3,1) both;
}
@keyframes cardIn { from{opacity:0;transform:translateY(24px) scale(.97)} to{opacity:1;transform:translateY(0) scale(1)} }

.login-logo {
  text-align: center;
  margin-bottom: 32px;
}
.login-logo-icon {
  width: 56px; height: 56px;
  background: rgba(0,212,255,.1);
  border: 1px solid rgba(0,212,255,.25);
  border-radius: 14px;
  display: flex; align-items: center; justify-content: center;
  font-size: 24px;
  margin: 0 auto 14px;
  box-shadow: 0 0 30px rgba(0,212,255,.15);
}
.login-logo-title {
  font-family: 'Syne', sans-serif;
  font-weight: 800;
  font-size: 20px;
  color: #e8f0fe;
  letter-spacing: -.3px;
}
.login-logo-sub {
  font-size: 10px;
  color: #3d5166;
  letter-spacing: 2px;
  text-transform: uppercase;
  margin-top: 4px;
}

.login-form { display: flex; flex-direction: column; gap: 16px; }

.login-field { display: flex; flex-direction: column; gap: 6px; }
.login-field label {
  font-size: 10px;
  color: #7a8fa6;
  text-transform: uppercase;
  letter-spacing: 1px;
}
.login-field input {
  background: #101820;
  border: 1px solid #1a2535;
  color: #e8f0fe;
  font-family: 'Space Mono', monospace;
  font-size: 13px;
  padding: 11px 14px;
  border-radius: 8px;
  outline: none;
  transition: border-color .2s, box-shadow .2s;
  width: 100%;
}
.login-field input:focus {
  border-color: #00d4ff;
  box-shadow: 0 0 0 3px rgba(0,212,255,.08);
}
.login-field input::placeholder { color: #3d5166; }

.login-error {
  background: rgba(255,56,96,.08);
  border: 1px solid rgba(255,56,96,.25);
  border-radius: 7px;
  padding: 10px 14px;
  font-size: 11px;
  color: #ff3860;
  text-align: center;
  animation: fadeIn .2s ease;
}
@keyframes fadeIn { from{opacity:0} to{opacity:1} }

.login-btn {
  background: #00d4ff;
  color: #060a0f;
  border: none;
  border-radius: 8px;
  padding: 13px;
  font-family: 'Syne', sans-serif;
  font-weight: 700;
  font-size: 13px;
  cursor: pointer;
  transition: all .15s;
  letter-spacing: .3px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  margin-top: 4px;
}
.login-btn:hover { background: #33ddff; box-shadow: 0 0 30px rgba(0,212,255,.4); }
.login-btn:disabled { opacity: .5; cursor: not-allowed; }
.login-btn .spinner { border-top-color: #060a0f; border-color: rgba(6,10,15,.3); border-top-color: #060a0f; }

.login-hints {
  margin-top: 24px;
  padding-top: 20px;
  border-top: 1px solid #1a2535;
}
.login-hints-title { font-size: 9px; color: #3d5166; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 10px; text-align: center; }
.login-hints-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 6px; }
.hint-card {
  background: #101820;
  border: 1px solid #1a2535;
  border-radius: 7px;
  padding: 9px 8px;
  cursor: pointer;
  transition: border-color .15s;
  text-align: center;
}
.hint-card:hover { border-color: #243044; }
.hint-role { font-size: 8px; color: #3d5166; text-transform: uppercase; letter-spacing: .5px; margin-bottom: 3px; }
.hint-user { font-size: 10px; color: #7a8fa6; font-family: 'Space Mono', monospace; }
`

const HINTS = [
  { role: 'Operator',   username: 'operator1',  password: 'operator_pass',  color: '#00d4ff' },
  { role: 'Scientist',  username: 'scientist1', password: 'scientist_pass', color: '#00ff9d' },
  { role: 'Admin',      username: 'admin',      password: 'admin_pass',     color: '#ff6b35' },
]

export default function Login() {
  const { login } = useAuth()
  const navigate  = useNavigate()
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error,    setError]    = useState('')
  const [loading,  setLoading]  = useState(false)

  async function handleSubmit(e) {
    e.preventDefault()
    if (!username || !password) { setError('Please enter username and password.'); return }
    setError('')
    setLoading(true)
    try {
      await login(username, password)
      navigate('/', { replace: true })
    } catch (err) {
      setError(err.message || 'Invalid credentials')
    } finally {
      setLoading(false)
    }
  }

  function fillHint(h) {
    setUsername(h.username)
    setPassword(h.password)
    setError('')
  }

  return (
    <>
      <style>{CSS}</style>
      <div className="login-root">
        <div className="login-grid" />
        <div className="login-orb login-orb-1" />
        <div className="login-orb login-orb-2" />
        <div className="login-orb login-orb-3" />

        <div className="login-card">
          <div className="login-logo">
            <div className="login-logo-icon">🗼</div>
            <div className="login-logo-title">5G Handover Platform</div>
            <div className="login-logo-sub">AI Optimization System</div>
          </div>

          <form className="login-form" onSubmit={handleSubmit}>
            <div className="login-field">
              <label htmlFor="username">Username</label>
              <input
                id="username"
                type="text"
                placeholder="Enter username"
                value={username}
                onChange={e => setUsername(e.target.value)}
                autoComplete="username"
                autoFocus
              />
            </div>
            <div className="login-field">
              <label htmlFor="password">Password</label>
              <input
                id="password"
                type="password"
                placeholder="Enter password"
                value={password}
                onChange={e => setPassword(e.target.value)}
                autoComplete="current-password"
              />
            </div>

            {error && <div className="login-error">{error}</div>}

            <button className="login-btn" type="submit" disabled={loading}>
              {loading
                ? <><span className="spinner" style={{width:14,height:14,borderWidth:2}} />Authenticating…</>
                : '→ Sign In'
              }
            </button>
          </form>

          <div className="login-hints">
            <div className="login-hints-title">Quick fill — test accounts</div>
            <div className="login-hints-grid">
              {HINTS.map(h => (
                <div key={h.username} className="hint-card" onClick={() => fillHint(h)}>
                  <div className="hint-role" style={{ color: h.color }}>{h.role}</div>
                  <div className="hint-user">{h.username}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </>
  )
}
