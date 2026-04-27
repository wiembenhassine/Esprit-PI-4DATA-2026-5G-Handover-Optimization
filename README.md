# 🗼 AI Handover Optimization Platform — v2 (Real Models)

Production FastAPI microservices platform for 5G handover prediction,
powered by **real trained models** from the DSO research pipeline.

---

## Real Model Artefacts

All 9 artefacts live in `model_registry/` and are loaded at startup:

| File | Type | DSO | Features |
|------|------|-----|----------|
| `scaler_dso1.pkl` | StandardScaler | DSO1 | 16 |
| `model_dso1_xgb.pkl` | XGBClassifier | DSO1 | 16 |
| `model_dso1_nn.keras` | Keras Sequential | DSO1 ensemble | 16 |
| `scaler_dso2.pkl` | StandardScaler | DSO2 | 7 |
| `model_dso2_xgb_honest.pkl` | XGBRegressor | DSO2 | 7 |
| `model_dso2_nn.keras` | Keras Sequential | DSO2 ensemble | 7 |
| `scaler_dso3.pkl` | StandardScaler | DSO3 | 7 |
| `model_dso3_kmeans.pkl` | KMeans (k=4) | DSO3 | 7 |
| `model_dso4_controller.pkl` | XGBClassifier | DSO4 | 18 |

---

## DSO Chain

```
Input Features
     │
     ▼
 ┌─────────────────────────────────────────────────────────┐
 │  DSO1 — Signal Degradation (XGBClassifier + Keras NN)  │
 │  Features: rsrp, rsrq, sinr, cqi, tx_power, ta,        │
 │            rsrp_delta_3, sinr_delta_3, velocity,         │
 │            num_neighbors, neighbor_gap, hour_of_day,     │
 │            day_of_week, cell_hist_datarate_mean,         │
 │            cell_load_drop_flag, latency_is_imputed       │
 │  Output:  degradation_prob + SHAP values                │
 └──────────────────────┬──────────────────────────────────┘
                        │  dso1_risk_score ──────────────────┐
                        ▼                                    │
 ┌────────────────────────────────────────────────────────┐ │
 │  DSO2 — Neighbor Gain (XGBRegressor + Keras NN)        │ │
 │  Features: rsrp, sinr, velocity, num_neighbors,        │ │
 │            mean_neighbor_rsrp, is_ho, hour_of_day      │ │
 │  Output:  predicted_neighbor_gap (dB)                  │ │
 └──────────────────────┬─────────────────────────────────┘ │
                        │  dso2_neighbor_gain ───────────────┤
                        ▼                                    │
 ┌────────────────────────────────────────────────────────┐ │
 │  DSO3 — User State (KMeans k=4)                        │ │
 │  Features: rsrp, sinr, datarate, velocity,             │ │
 │            hour_of_day, day_of_week,                   │ │
 │            cell_hist_datarate_mean                     │ │
 │  Clusters: 0=good, 1=fair, 2=cell_edge, 3=congested   │ │
 └──────────────────────┬─────────────────────────────────┘ │
                        │  dso3_cluster ─────────────────────┤
                        ▼                                    │
 ┌────────────────────────────────────────────────────────┐ │
 │  DSO4 — Master Controller (XGBClassifier)    ◄─────────┘ │
 │  Features: base 15 + dso1_risk_score +                 │ │
 │            dso2_neighbor_gain + dso3_cluster           │ │
 │  Output:  handover_recommended + confidence            │ │
 └────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Option A — Local Python

```bash
unzip 5g_handover_platform.zip
cd 5g_handover_platform
pip install -r requirements.txt
./run_all.sh
```

Open **http://localhost:8000/docs**

### Option B — Docker Compose

```bash
unzip 5g_handover_platform.zip
cd 5g_handover_platform
docker compose up --build
```

---

## API Usage

### 1. Authenticate

```bash
curl -X POST http://localhost:8000/auth/token \
  -d "username=operator1&password=operator_pass"
# → { "access_token": "eyJ...", "role": "network_operator" }
TOKEN="eyJ..."
```

### 2. Minimal Prediction (3 fields)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": 1700000000,
    "scenario": "hbahn",
    "rsrp": -98,
    "rsrq": -14,
    "sinr": 4
  }'
```

### 3. Full Prediction (all fields)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": 1700000000,
    "scenario": "hbahn",
    "rsrp": -98.0,   "rsrq": -14.0,   "sinr": 4.0,
    "cqi": 7,        "tx_power": 21.0, "ta": 6,
    "velocity": 80.0,
    "n_neighbors": 3, "neighbor_gap": 5.5, "best_neighbor_rsrp": -92.5,
    "rsrp_slope3": -3.0, "sinr_slope3": -1.0,
    "hour_of_day": 8, "day_of_week": 1,
    "cell_hist_datarate_mean": 10.0, "cell_load_drop_flag": 0,
    "latency_is_imputed": 0
  }'
```

### 4. Batch Prediction

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '[
    {"timestamp":1700000000,"scenario":"hbahn","rsrp":-90,"rsrq":-11,"sinr":8},
    {"timestamp":1700000010,"scenario":"mobile","rsrp":-105,"rsrq":-16,"sinr":1}
  ]'
```

---

## Enable Keras NN Ensemble

By default, only XGBoost is used. To enable XGB+NN averaging:

```bash
export KERAS_ENSEMBLE=true
./run_all.sh
```

First inference per session is ~1.5s (Keras JIT compile); subsequent calls ~280ms.

---

## Service Ports

| Service | Port | Swagger |
|---------|------|---------|
| API Gateway | 8000 | http://localhost:8000/docs |
| Data Ingestion | 8001 | http://localhost:8001/docs |
| Data Processing | 8002 | http://localhost:8002/docs |
| **Prediction (real models)** | **8003** | http://localhost:8003/docs |
| Monitoring | 8004 | http://localhost:8004/docs |
| MLOps | 8005 | http://localhost:8005/docs |
| Dashboard | 8006 | http://localhost:8006/docs |
| Auth | 8007 | http://localhost:8007/docs |

---

## Frontend

The React frontend (`5g_dashboard.jsx`) connects to the API Gateway.
Change line 8 from `MOCK = true` to `MOCK = false` to use live predictions.

## Test Credentials

| User | Password | Role |
|------|----------|------|
| operator1 | operator_pass | network_operator |
| scientist1 | scientist_pass | data_scientist |
| admin | admin_pass | admin |
