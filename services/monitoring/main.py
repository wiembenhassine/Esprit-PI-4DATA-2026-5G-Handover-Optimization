"""
Monitoring Service — Port 8004
Monitors model performance in production, detects concept drift and anomalies,
and fires alerts when thresholds are breached.

Drift detection: Population Stability Index (PSI) on feature distributions.
Anomaly detection: Z-score on rolling prediction distributions.
"""
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import deque
import numpy as np
import uuid
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from shared.schemas import (
    ModelMetrics, Alert, DriftStatus,
    UserInfo, UserRole
)
from shared.config import settings
from services.auth.main import get_current_user, require_role

app = FastAPI(
    title="Monitoring Service",
    description="Monitors model performance and detects concept drift and anomalies.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Drift Detection ───────────────────────────────────────────────────────────

def compute_psi(baseline: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """
    Population Stability Index.
    PSI < 0.1: no drift | 0.1–0.2: moderate | > 0.2: significant
    """
    eps = 1e-8
    bins   = np.percentile(baseline, np.linspace(0, 100, n_bins + 1))
    bins[0] -= 1; bins[-1] += 1
    base_freq = np.histogram(baseline, bins=bins)[0] / len(baseline) + eps
    curr_freq = np.histogram(current,  bins=bins)[0] / len(current)  + eps
    psi = np.sum((curr_freq - base_freq) * np.log(curr_freq / base_freq))
    return float(np.clip(psi, 0, 5))


# ── State ─────────────────────────────────────────────────────────────────────

# Sliding window of recent predictions per model
prediction_windows: Dict[str, deque] = {
    "dso1": deque(maxlen=settings.MONITORING_WINDOW_SIZE),
    "dso2": deque(maxlen=settings.MONITORING_WINDOW_SIZE),
    "dso3": deque(maxlen=settings.MONITORING_WINDOW_SIZE),
    "dso4": deque(maxlen=settings.MONITORING_WINDOW_SIZE),
}

# Baseline distributions (populated on first window fill)
baselines: Dict[str, Optional[np.ndarray]] = {
    "dso1": None, "dso2": None, "dso3": None, "dso4": None,
}

metrics_history: List[ModelMetrics] = []
alerts_store:    List[Alert]        = []


def classify_drift(psi: float) -> DriftStatus:
    if psi > settings.DRIFT_CRITICAL_THRESHOLD * 2:
        return DriftStatus.DRIFT
    elif psi > settings.DRIFT_WARNING_THRESHOLD:
        return DriftStatus.WARNING
    return DriftStatus.OK


def fire_alert(model_name: str, metric: str, value: float, threshold: float, severity: str) -> Alert:
    alert = Alert(
        alert_id=str(uuid.uuid4()),
        severity=severity,
        model_name=model_name,
        message=f"{model_name} — {metric} = {value:.4f} exceeds threshold {threshold:.4f}",
        metric=metric,
        value=value,
        threshold=threshold,
    )
    alerts_store.append(alert)
    print(f"[Monitor] 🚨 ALERT [{severity.upper()}]: {alert.message}")
    return alert


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/monitor/record", tags=["Monitoring"])
async def record_prediction(
    payload: Dict[str, Any],
    current_user: UserInfo = Depends(get_current_user),
):
    """
    Record a prediction result for monitoring.
    Called internally by the Prediction Service after each inference.

    Expected payload keys: dso1_prob, dso2_gap, dso3_cluster, dso4_prob
    """
    dso1_prob   = float(payload.get("dso1_prob",   0.5))
    dso2_gap    = float(payload.get("dso2_gap",    0.0))
    dso3_cluster = int(payload.get("dso3_cluster", 0))
    dso4_prob   = float(payload.get("dso4_prob",   0.5))

    prediction_windows["dso1"].append(dso1_prob)
    prediction_windows["dso2"].append(dso2_gap)
    prediction_windows["dso3"].append(float(dso3_cluster))
    prediction_windows["dso4"].append(dso4_prob)

    # Establish baseline on first fill
    for name, window in prediction_windows.items():
        if len(window) == settings.MONITORING_WINDOW_SIZE and baselines[name] is None:
            baselines[name] = np.array(window)
            print(f"[Monitor] Baseline established for {name} ({len(window)} samples)")

    return {"recorded": True, "window_sizes": {k: len(v) for k, v in prediction_windows.items()}}


@app.get("/monitor/drift", tags=["Monitoring"])
async def check_drift(current_user: UserInfo = Depends(get_current_user)):
    """
    Compute PSI drift scores for all models against their baselines.
    Fires alerts if thresholds are breached.
    """
    results = {}
    for name, window in prediction_windows.items():
        if len(window) < 50:
            results[name] = {"status": "insufficient_data", "n": len(window)}
            continue
        baseline = baselines[name]
        if baseline is None:
            results[name] = {"status": "no_baseline", "n": len(window)}
            continue

        current = np.array(window)
        psi     = compute_psi(baseline, current)
        status  = classify_drift(psi)

        if status == DriftStatus.DRIFT:
            fire_alert(name, "PSI", psi, settings.DRIFT_CRITICAL_THRESHOLD * 2, "critical")
        elif status == DriftStatus.WARNING:
            fire_alert(name, "PSI", psi, settings.DRIFT_WARNING_THRESHOLD, "warning")

        results[name] = {
            "psi":         round(psi, 4),
            "drift_status": status,
            "n_current":   len(window),
            "n_baseline":  len(baseline),
        }
    return results


@app.get("/monitor/metrics", response_model=List[ModelMetrics], tags=["Monitoring"])
async def get_metrics(
    model_name: Optional[str] = None,
    current_user: UserInfo = Depends(get_current_user),
):
    """Return recent model performance metrics."""
    # Compute live metrics from windows
    live = []
    for name, window in prediction_windows.items():
        if not window:
            continue
        arr = np.array(window)
        baseline = baselines[name]
        psi      = compute_psi(baseline, arr) if baseline is not None else 0.0
        drift    = classify_drift(psi)
        live.append(ModelMetrics(
            model_name=name,
            # Classification metrics (from ground-truth labels if available)
            accuracy=None,  # populated when ground-truth labels are joined
            f1_score=None,
            roc_auc=None,
            n_predictions=len(window),
            drift_status=drift,
            drift_score=round(psi, 4),
        ))
    if model_name:
        live = [m for m in live if m.model_name == model_name]
    return live


@app.post("/monitor/evaluate", tags=["Monitoring"])
async def evaluate_model(
    payload: Dict[str, Any],
    current_user: UserInfo = Depends(require_role(UserRole.DATA_SCIENTIST, UserRole.ADMIN)),
):
    """
    Submit ground-truth labels to evaluate model performance.
    Data Scientist / Admin only.

    Payload: { "model": "dso1", "y_true": [...], "y_pred": [...] }
    """
    model   = payload.get("model", "dso1")
    y_true  = np.array(payload.get("y_true", []))
    y_pred  = np.array(payload.get("y_pred", []))

    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise HTTPException(status_code=422, detail="y_true and y_pred must be equal-length non-empty arrays.")

    if model in ("dso1", "dso4"):
        # Binary classification metrics
        tp = int(np.sum((y_true == 1) & (y_pred >= 0.5)))
        tn = int(np.sum((y_true == 0) & (y_pred < 0.5)))
        fp = int(np.sum((y_true == 0) & (y_pred >= 0.5)))
        fn = int(np.sum((y_true == 1) & (y_pred < 0.5)))
        acc  = (tp + tn) / len(y_true)
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        metrics = {"accuracy": round(acc, 4), "precision": round(prec, 4),
                   "recall": round(rec, 4), "f1_score": round(f1, 4)}
    else:
        # Regression metrics (DSO2)
        mse = float(np.mean((y_true - y_pred) ** 2))
        mae = float(np.mean(np.abs(y_true - y_pred)))
        metrics = {"mse": round(mse, 4), "mae": round(mae, 4),
                   "rmse": round(float(np.sqrt(mse)), 4)}

    result = ModelMetrics(
        model_name=model,
        n_predictions=len(y_true),
        drift_status=DriftStatus.OK,
        drift_score=0.0,
        **{k: v for k, v in metrics.items() if k in ModelMetrics.model_fields},
    )
    metrics_history.append(result)
    return {"model": model, "n_samples": len(y_true), "metrics": metrics}


@app.get("/monitor/alerts", response_model=List[Alert], tags=["Monitoring"])
async def get_alerts(
    severity: Optional[str] = None,
    limit: int = 50,
    current_user: UserInfo = Depends(get_current_user),
):
    """Retrieve recent monitoring alerts."""
    items = alerts_store[-limit:]
    if severity:
        items = [a for a in items if a.severity == severity]
    return items


@app.delete("/monitor/alerts", tags=["Monitoring"])
async def clear_alerts(
    admin: UserInfo = Depends(require_role(UserRole.ADMIN)),
):
    """Clear all alerts. Admin only."""
    alerts_store.clear()
    return {"cleared": True}


@app.get("/health", tags=["Health"])
async def health():
    return {
        "service": "monitoring",
        "status":  "healthy",
        "windows": {k: len(v) for k, v in prediction_windows.items()},
        "timestamp": datetime.utcnow(),
    }
