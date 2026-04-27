// api.js — all calls go through /api proxy → http://localhost:8000
const BASE = '/api'

function getToken() {
  return localStorage.getItem('token')
}

function authHeaders() {
  return {
    'Content-Type': 'application/json',
    Authorization: `Bearer ${getToken()}`,
  }
}

async function request(method, path, body) {
  const res = await fetch(BASE + path, {
    method,
    headers: authHeaders(),
    body: body ? JSON.stringify(body) : undefined,
  })
  if (res.status === 401) {
    localStorage.removeItem('token')
    localStorage.removeItem('user')
    window.location.href = '/login'
    return
  }
  const data = await res.json()
  if (!res.ok) throw new Error(data?.detail || `HTTP ${res.status}`)
  return data
}

// ── Auth ─────────────────────────────────────────────────────────────────────

export async function login(username, password) {
  const form = new URLSearchParams({ username, password })
  const res = await fetch(`${BASE}/auth/token`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: form,
  })
  const data = await res.json()
  if (!res.ok) throw new Error(data?.detail || 'Login failed')
  return data // { access_token, token_type, expires_in, role }
}

export async function getMe() {
  return request('GET', '/auth/me')
}

// ── Dashboard ─────────────────────────────────────────────────────────────────

export async function getKPIs() {
  return request('GET', '/dashboard/kpis')
}

export async function getLiveFeed(limit = 20) {
  return request('GET', `/dashboard/live-feed?limit=${limit}`)
}

export async function getScenarios() {
  return request('GET', '/dashboard/scenarios')
}

// ── Prediction ────────────────────────────────────────────────────────────────

export async function predict(features) {
  return request('POST', '/predict', features)
}

export async function predictBatch(list) {
  return request('POST', '/predict/batch', list)
}

export async function getPredictStats() {
  return request('GET', '/predict/stats')
}

export async function getPredictHistory(limit = 50, scenario) {
  const q = scenario ? `?limit=${limit}&scenario=${scenario}` : `?limit=${limit}`
  return request('GET', `/predict/history${q}`)
}

export async function getModels() {
  return request('GET', '/predict/models')
}

// ── Monitoring ────────────────────────────────────────────────────────────────

export async function getDrift() {
  return request('GET', '/monitor/drift')
}

export async function getAlerts(severity, limit = 50) {
  const q = severity ? `?severity=${severity}&limit=${limit}` : `?limit=${limit}`
  return request('GET', `/monitor/alerts${q}`)
}

export async function getMetrics(model) {
  const q = model ? `?model_name=${model}` : ''
  return request('GET', `/monitor/metrics${q}`)
}

// ── MLOps ─────────────────────────────────────────────────────────────────────

export async function getRegistry() {
  return request('GET', '/mlops/registry')
}

export async function triggerTraining(model_name, dataset_path = '/data/train.csv', hyperparams = {}) {
  return request('POST', '/mlops/train', { model_name, dataset_path, hyperparams })
}

export async function getJobs() {
  return request('GET', '/mlops/jobs')
}

export async function deployModel(model_name, version) {
  return request('POST', `/mlops/registry/${model_name}/${version}/deploy`)
}

// ── Health ────────────────────────────────────────────────────────────────────

export async function getHealth() {
  return request('GET', '/health')
}
