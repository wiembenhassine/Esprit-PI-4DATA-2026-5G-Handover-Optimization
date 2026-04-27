"""
Prediction Service — Port 8003
Real trained models: DSO1(XGB+NN) → DSO2(XGB+NN) → DSO3(KMeans) → DSO4(XGB)
"""
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from typing import List, Optional, Dict, Any
import os, warnings, time, uuid, logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
import joblib
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from shared.schemas import (
    EngineeredFeatures, PredictionResult,
    DSO1Output, DSO2Output, DSO3Output, DSO4Output, UserInfo,
)
from shared.config import settings
from services.auth.main import get_current_user

app = FastAPI(title="Prediction Service (DSO1-DSO4)", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=settings.ALLOWED_ORIGINS,
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

REGISTRY = os.path.join(os.path.dirname(__file__), "../../model_registry")
CLUSTER_LABELS = {0: "good", 1: "fair", 2: "cell_edge", 3: "congested"}

DSO1_FEATURES = ["rsrp","rsrq","sinr","cqi","tx_power","ta","rsrp_delta_3","sinr_delta_3",
                 "velocity","num_neighbors","neighbor_gap","hour_of_day","day_of_week",
                 "cell_hist_datarate_mean","cell_load_drop_flag","latency_is_imputed"]
DSO2_FEATURES = ["rsrp","sinr","velocity","num_neighbors","mean_neighbor_rsrp","is_ho","hour_of_day"]
DSO3_FEATURES = ["rsrp","sinr","datarate","velocity","hour_of_day","day_of_week","cell_hist_datarate_mean"]
DSO4_FEATURES = ["rsrp","rsrq","sinr","cqi","tx_power","ta","velocity","num_neighbors",
                 "best_neighbor_rsrp","neighbor_gap","hour_of_day","day_of_week",
                 "cell_hist_datarate_mean","cell_load_drop_flag","latency_is_imputed",
                 "dso3_cluster","dso1_risk_score","dso2_neighbor_gain"]

class _Models:
    loaded = False
    def load(self):
        if self.loaded:
            return
        import tensorflow as tf
        p = REGISTRY
        print(f"[Prediction] Loading models from {p}")
        self.scaler_dso1    = joblib.load(f"{p}/scaler_dso1.pkl")
        self.scaler_dso2    = joblib.load(f"{p}/scaler_dso2.pkl")
        self.scaler_dso3    = joblib.load(f"{p}/scaler_dso3.pkl")
        self.model_dso1_xgb = joblib.load(f"{p}/model_dso1_xgb.pkl")
        self.model_dso1_nn  = tf.keras.models.load_model(f"{p}/model_dso1_nn.keras", compile=False)
        self.model_dso2_xgb = joblib.load(f"{p}/model_dso2_xgb_honest.pkl")
        self.model_dso2_nn  = tf.keras.models.load_model(f"{p}/model_dso2_nn.keras", compile=False)
        self.model_dso3     = joblib.load(f"{p}/model_dso3_kmeans.pkl")
        self.model_dso4     = joblib.load(f"{p}/model_dso4_controller.pkl")
        self.loaded = True
        print("[Prediction] All 9 artefacts loaded.")

M = _Models()

@app.on_event("startup")
async def startup():
    M.load()

def _v(val, default):
    return val if val is not None else default

def _row1(f):
    return pd.DataFrame([{
        "rsrp": f.rsrp, "rsrq": f.rsrq, "sinr": f.sinr,
        "cqi": _v(f.cqi, 10), "tx_power": _v(f.tx_power, 15.0), "ta": _v(f.ta, 3.0),
        "rsrp_delta_3": _v(f.rsrp_slope3, 0.0), "sinr_delta_3": _v(f.sinr_slope3, 0.0),
        "velocity": _v(f.velocity, 0.0), "num_neighbors": _v(f.n_neighbors, 0),
        "neighbor_gap": _v(f.neighbor_gap, 0.0), "hour_of_day": _v(f.hour_of_day, 0),
        "day_of_week": _v(f.day_of_week, 0),
        "cell_hist_datarate_mean": _v(f.cell_hist_datarate_mean, 25.0),
        "cell_load_drop_flag": _v(f.cell_load_drop_flag, 0),
        "latency_is_imputed": _v(f.latency_is_imputed, 1),
    }])[DSO1_FEATURES]

def _row2(f):
    mnr = _v(f.best_neighbor_rsrp, f.rsrp + 5.0)
    return pd.DataFrame([{
        "rsrp": f.rsrp, "sinr": f.sinr, "velocity": _v(f.velocity, 0.0),
        "num_neighbors": _v(f.n_neighbors, 0), "mean_neighbor_rsrp": mnr,
        "is_ho": 0, "hour_of_day": _v(f.hour_of_day, 0),
    }])[DSO2_FEATURES]

def _row3(f):
    return pd.DataFrame([{
        "rsrp": f.rsrp, "sinr": f.sinr,
        "datarate": _v(f.cell_hist_datarate_mean, 25.0),
        "velocity": _v(f.velocity, 0.0), "hour_of_day": _v(f.hour_of_day, 0),
        "day_of_week": _v(f.day_of_week, 0),
        "cell_hist_datarate_mean": _v(f.cell_hist_datarate_mean, 25.0),
    }])[DSO3_FEATURES]

def _row4(f, risk, gain, cluster):
    return pd.DataFrame([{
        "rsrp": f.rsrp, "rsrq": f.rsrq, "sinr": f.sinr,
        "cqi": _v(f.cqi, 10), "tx_power": _v(f.tx_power, 15.0), "ta": _v(f.ta, 3.0),
        "velocity": _v(f.velocity, 0.0), "num_neighbors": _v(f.n_neighbors, 0),
        "best_neighbor_rsrp": _v(f.best_neighbor_rsrp, f.rsrp + gain),
        "neighbor_gap": _v(f.neighbor_gap, gain),
        "hour_of_day": _v(f.hour_of_day, 0), "day_of_week": _v(f.day_of_week, 0),
        "cell_hist_datarate_mean": _v(f.cell_hist_datarate_mean, 25.0),
        "cell_load_drop_flag": _v(f.cell_load_drop_flag, 0),
        "latency_is_imputed": _v(f.latency_is_imputed, 1),
        "dso3_cluster": cluster, "dso1_risk_score": risk, "dso2_neighbor_gain": gain,
    }])[DSO4_FEATURES]

def _shap(X):
    try:
        import xgboost as xgb
        dm = xgb.DMatrix(X, feature_names=DSO1_FEATURES)
        c = M.model_dso1_xgb.get_booster().predict(dm, pred_contribs=True)[0, :-1]
        top = np.argsort(np.abs(c))[::-1][:6]
        return {DSO1_FEATURES[i]: round(float(c[i]), 5) for i in top}
    except Exception:
        return {}

def run_pipeline(f: EngineeredFeatures):
    X1 = M.scaler_dso1.transform(_row1(f))
    risk = round(float(np.mean([
        M.model_dso1_xgb.predict_proba(X1)[0, 1],
        float(M.model_dso1_nn.predict(X1, verbose=0)[0, 0])
    ])), 4)
    dso1 = DSO1Output(
        is_degrading=risk > 0.5, degradation_prob=risk,
        rsrp_future_5_pred=round(float(f.rsrp + 5 * _v(f.rsrp_slope3, _v(f.rsrp_lag1, f.rsrp) - f.rsrp)), 2),
        shap_values=_shap(X1), model_version="xgb+nn_v2",
    )

    X2 = M.scaler_dso2.transform(_row2(f))
    gain = round(float(np.mean([
        M.model_dso2_xgb.predict(X2)[0],
        float(M.model_dso2_nn.predict(X2, verbose=0)[0, 0])
    ])), 4)
    dso2 = DSO2Output(
        predicted_neighbor_gap=gain,
        predicted_best_rsrp=round(float(f.rsrp + gain), 2),
        model_version="xgb_honest+nn_v2",
    )

    X3 = M.scaler_dso3.transform(_row3(f))
    cluster = int(M.model_dso3.predict(X3)[0])
    dists = np.linalg.norm(M.model_dso3.cluster_centers_ - X3[0], axis=1)
    sc = 1.0 / (dists + 1e-8)
    probs = {int(k): round(float(v / sc.sum()), 4) for k, v in enumerate(sc)}
    dso3 = DSO3Output(
        cluster_id=cluster, cluster_label=CLUSTER_LABELS.get(cluster, f"cluster_{cluster}"),
        cluster_probs=probs, model_version="kmeans_v2",
    )

    X4 = _row4(f, risk, gain, cluster)
    ho = round(float(M.model_dso4.predict_proba(X4)[0, 1]), 4)
    dso4 = DSO4Output(
        handover_recommended=ho > 0.5, handover_prob=ho,
        confidence=("high" if abs(ho-0.5)>0.25 else "medium" if abs(ho-0.5)>0.10 else "low"),
        model_version="xgb_controller_v2",
    )
    return dso1, dso2, dso3, dso4

prediction_history: List[PredictionResult] = []

@app.post("/predict", response_model=PredictionResult, tags=["Prediction"])
async def predict(features: EngineeredFeatures, current_user: UserInfo = Depends(get_current_user)):
    t0 = time.time()
    try:
        dso1, dso2, dso3, dso4 = run_pipeline(features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")
    r = PredictionResult(request_id=str(uuid.uuid4()), timestamp=features.timestamp,
        scenario=features.scenario, dso1=dso1, dso2=dso2, dso3=dso3, dso4=dso4,
        latency_ms=round((time.time()-t0)*1000, 2))
    prediction_history.append(r)
    return r

@app.post("/predict/batch", response_model=List[PredictionResult], tags=["Prediction"])
async def predict_batch(feature_list: List[EngineeredFeatures], current_user: UserInfo = Depends(get_current_user)):
    if len(feature_list) > 1000:
        raise HTTPException(status_code=400, detail="Batch max 1000")
    results = []
    for f in feature_list:
        t0 = time.time()
        dso1, dso2, dso3, dso4 = run_pipeline(f)
        r = PredictionResult(request_id=str(uuid.uuid4()), timestamp=f.timestamp,
            scenario=f.scenario, dso1=dso1, dso2=dso2, dso3=dso3, dso4=dso4,
            latency_ms=round((time.time()-t0)*1000, 2))
        results.append(r); prediction_history.append(r)
    return results

@app.get("/predict/history", tags=["Prediction"])
async def get_history(limit: int = 100, scenario: Optional[str] = None,
                      current_user: UserInfo = Depends(get_current_user)):
    items = prediction_history[-limit:]
    if scenario: items = [p for p in items if p.scenario == scenario]
    return {"total": len(prediction_history), "returned": len(items), "items": items}

@app.get("/predict/stats", tags=["Prediction"])
async def stats(current_user: UserInfo = Depends(get_current_user)):
    if not prediction_history: return {"message": "No predictions yet."}
    ho = [p.dso4.handover_recommended for p in prediction_history]
    deg = [p.dso1.is_degrading for p in prediction_history]
    lats = [p.latency_ms for p in prediction_history]
    clust = {}
    for p in prediction_history:
        clust[p.dso3.cluster_label] = clust.get(p.dso3.cluster_label, 0) + 1
    return {"total_predictions": len(prediction_history),
            "handover_rate": round(sum(ho)/len(ho), 4),
            "degradation_rate": round(sum(deg)/len(deg), 4),
            "avg_latency_ms": round(sum(lats)/len(lats), 2),
            "p95_latency_ms": round(sorted(lats)[int(len(lats)*0.95)], 2),
            "cluster_distribution": clust}

@app.get("/predict/models", tags=["Prediction"])
async def list_models(current_user: UserInfo = Depends(get_current_user)):
    return {"models": [
        {"dso":"DSO1","ensemble":["XGBClassifier","Keras NN"],"features":DSO1_FEATURES},
        {"dso":"DSO2","ensemble":["XGBRegressor(honest)","Keras NN"],"features":DSO2_FEATURES},
        {"dso":"DSO3","k":4,"cluster_labels":CLUSTER_LABELS,"features":DSO3_FEATURES},
        {"dso":"DSO4","model":"XGBClassifier","chained":["dso1_risk_score","dso2_neighbor_gain","dso3_cluster"],"features":DSO4_FEATURES},
    ]}

@app.get("/health", tags=["Health"])
async def health():
    return {"service":"prediction","status":"healthy","models_loaded":M.loaded,"timestamp":datetime.utcnow()}
