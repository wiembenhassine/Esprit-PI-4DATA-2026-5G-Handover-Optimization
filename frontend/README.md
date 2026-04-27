# 5G Handover Platform — Frontend

React + Vite frontend for the AI Handover Optimization Platform.
Full login, protected routes, real API calls to the FastAPI backend.

## Pages

| Page | Route | Access |
|------|-------|--------|
| **Login** | `/login` | Public |
| **Overview** | `/` | All roles |
| **Predict** | `/predict` | All roles |
| **History** | `/history` | All roles |
| **Drift & Alerts** | `/monitoring` | All roles |
| **MLOps** | `/mlops` | data_scientist + admin |

## Quick Start

### 1 — Make sure the backend is running

```bash
cd ..                   # project root
./run_all.sh            # starts all 8 FastAPI services
```

### 2 — Install and start the frontend

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173**

### 3 — Login

Use any of the test accounts:

| Username | Password | Role |
|----------|----------|------|
| operator1 | operator_pass | Network Operator |
| scientist1 | scientist_pass | Data Scientist |
| admin | admin_pass | Admin |

Or click the quick-fill buttons on the login page.

## How the proxy works

Vite proxies all `/api/*` calls to `http://localhost:8000` (the FastAPI gateway).
This avoids CORS issues in development. In production, configure your web server
(nginx, Caddy) to do the same proxying.

## Build for production

```bash
npm run build
# Output is in dist/ — serve with any static file server
```

## Project structure

```
frontend/
├── index.html
├── vite.config.js        ← proxy /api → localhost:8000
├── package.json
└── src/
    ├── main.jsx           ← React root
    ├── App.jsx            ← Router + protected routes
    ├── AuthContext.jsx    ← Login state, token, role helpers
    ├── api.js             ← All backend API calls
    ├── styles.css         ← Global design tokens + shared classes
    ├── components/
    │   └── Layout.jsx     ← Sidebar + topbar shell
    └── pages/
        ├── Login.jsx      ← Auth form + quick-fill hints
        ├── Overview.jsx   ← Live KPIs, signal chart, feed table
        ├── Predict.jsx    ← Full DSO form + result display + SHAP
        ├── History.jsx    ← Searchable prediction history
        ├── Monitoring.jsx ← PSI drift gauges + radar + alerts
        └── MLOps.jsx      ← Model registry + retrain + deploy
```
