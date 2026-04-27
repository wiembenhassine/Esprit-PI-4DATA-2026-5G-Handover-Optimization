"""
Dashboard Service — Port 8006
Aggregates predictions, alerts, and insights for the Network Operator UI.
All inter-service calls include the internal service token.
"""
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from typing import List, Dict, Any, Optional
import httpx
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from shared.schemas import UserInfo, UserRole
from shared.config import settings
from services.auth.main import get_current_user

app = FastAPI(
    title="Dashboard Service",
    description="Aggregates predictions, alerts, and insights for network operator dashboards.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Internal HTTP client (always sends service token) ─────────────────────────

def _internal_headers() -> dict:
    return {"Authorization": f"Bearer {settings.INTERNAL_SERVICE_TOKEN}"}


async def fetch(url: str, path: str) -> Optional[Dict]:
    """GET an internal service endpoint with service-to-service auth."""
    async with httpx.AsyncClient(timeout=8.0) as client:
        try:
            r = await client.get(f"{url}{path}", headers=_internal_headers())
            if r.status_code == 200:
                return r.json()
            return None
        except Exception as e:
            print(f"[Dashboard] fetch failed {url}{path}: {e}")
            return None


async def post(url: str, path: str, payload: dict) -> Optional[Dict]:
    """POST to an internal service endpoint with service-to-service auth."""
    async with httpx.AsyncClient(timeout=8.0) as client:
        try:
            r = await client.post(f"{url}{path}", json=payload, headers=_internal_headers())
            if r.status_code == 200:
                return r.json()
            return None
        except Exception as e:
            print(f"[Dashboard] post failed {url}{path}: {e}")
            return None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/dashboard/overview", tags=["Dashboard"])
async def overview(current_user: UserInfo = Depends(get_current_user)):
    pred_stats   = await fetch(settings.PREDICTION_URL,   "/predict/stats")
    ingest_stats = await fetch(settings.DATA_INGESTION_URL, "/ingest/stats")
    alerts       = await fetch(settings.MONITORING_URL,   "/monitor/alerts?limit=5")
    drift        = await fetch(settings.MONITORING_URL,   "/monitor/drift")
    registry     = await fetch(settings.MLOPS_URL,        "/mlops/registry")

    return {
        "generated_at":  datetime.utcnow().isoformat(),
        "ingestion":     ingest_stats or {"status": "unavailable"},
        "predictions":   pred_stats   or {"status": "unavailable"},
        "recent_alerts": (alerts or [])[:5],
        "drift_status":  drift or {},
        "active_models": {
            name: next(
                (v["version"] for v in versions if v["status"] == "deployed"), "none"
            )
            for name, versions in (registry or {}).items()
        },
    }


@app.get("/dashboard/live-feed", tags=["Dashboard"])
async def live_feed(
    limit: int = 20,
    current_user: UserInfo = Depends(get_current_user),
):
    history = await fetch(settings.PREDICTION_URL, f"/predict/history?limit={limit}")
    if not history:
        return {"items": [], "total_available": 0, "returned": 0}

    items = history.get("items", [])
    enriched = []
    for p in items:
        dso4 = p.get("dso4", {})
        dso1 = p.get("dso1", {})
        dso3 = p.get("dso3", {})
        enriched.append({
            "request_id":    p.get("request_id"),
            "timestamp":     p.get("timestamp"),
            "scenario":      p.get("scenario"),
            "handover":      dso4.get("handover_recommended", False),
            "handover_prob": dso4.get("handover_prob", 0),
            "confidence":    dso4.get("confidence", "low"),
            "degrading":     dso1.get("is_degrading", False),
            "risk_score":    dso1.get("degradation_prob", 0),
            "network_state": dso3.get("cluster_label", "unknown"),
            "rsrp_future":   dso1.get("rsrp_future_5_pred"),
            "latency_ms":    p.get("latency_ms"),
        })
    return {
        "total_available": history.get("total", 0),
        "returned":        len(enriched),
        "items":           enriched,
    }


@app.get("/dashboard/kpis", tags=["Dashboard"])
async def kpis(current_user: UserInfo = Depends(get_current_user)):
    stats = await fetch(settings.PREDICTION_URL, "/predict/stats")
    drift = await fetch(settings.MONITORING_URL, "/monitor/drift")

    if not stats or "message" in stats:
        return {
            "total_predictions": 0,
            "handover_rate_pct": 0,
            "degradation_rate_pct": 0,
            "avg_inference_latency_ms": 0,
            "p95_inference_latency_ms": 0,
            "network_state_distribution": {},
            "system_health": "healthy",
            "max_drift_psi": 0,
        }

    drift_scores = {
        k: v.get("psi", 0)
        for k, v in (drift or {}).items()
        if isinstance(v, dict)
    }
    max_drift = max(drift_scores.values()) if drift_scores else 0

    return {
        "total_predictions":         stats.get("total_predictions", 0),
        "handover_rate_pct":         round(stats.get("handover_rate", 0) * 100, 2),
        "degradation_rate_pct":      round(stats.get("degradation_rate", 0) * 100, 2),
        "avg_inference_latency_ms":  stats.get("avg_latency_ms", 0),
        "p95_inference_latency_ms":  stats.get("p95_latency_ms", 0),
        "network_state_distribution": stats.get("cluster_distribution", {}),
        "max_drift_psi":             round(max_drift, 4),
        "system_health": (
            "critical" if max_drift > 0.20
            else "warning" if max_drift > 0.10
            else "healthy"
        ),
    }


@app.get("/dashboard/alerts", tags=["Dashboard"])
async def get_alerts(
    severity: Optional[str] = None,
    limit: int = 50,
    current_user: UserInfo = Depends(get_current_user),
):
    url = f"/monitor/alerts?limit={limit}"
    if severity:
        url += f"&severity={severity}"
    return await fetch(settings.MONITORING_URL, url) or []


@app.get("/dashboard/shap/{request_id}", tags=["Dashboard"])
async def get_shap(
    request_id: str,
    current_user: UserInfo = Depends(get_current_user),
):
    history = await fetch(settings.PREDICTION_URL, "/predict/history?limit=500")
    if not history:
        raise HTTPException(status_code=503, detail="Prediction service unavailable")
    prediction = next(
        (p for p in history.get("items", []) if p["request_id"] == request_id), None
    )
    if not prediction:
        raise HTTPException(status_code=404, detail=f"Prediction {request_id} not found")
    shap = prediction.get("dso1", {}).get("shap_values") or {}
    return {
        "request_id":  request_id,
        "shap_values": shap,
        "prediction": {
            "is_degrading":     prediction["dso1"]["is_degrading"],
            "degradation_prob": prediction["dso1"]["degradation_prob"],
        },
    }


@app.get("/dashboard/scenarios", tags=["Dashboard"])
async def scenario_summary(current_user: UserInfo = Depends(get_current_user)):
    history = await fetch(settings.PREDICTION_URL, "/predict/history?limit=1000")
    if not history:
        return {}
    from collections import defaultdict
    sc_data = defaultdict(lambda: {"count": 0, "handovers": 0, "degradations": 0})
    for p in history.get("items", []):
        sc = p.get("scenario", "unknown")
        sc_data[sc]["count"] += 1
        if p.get("dso4", {}).get("handover_recommended"):
            sc_data[sc]["handovers"] += 1
        if p.get("dso1", {}).get("is_degrading"):
            sc_data[sc]["degradations"] += 1
    return {
        sc: {
            **data,
            "handover_rate_pct":    round(data["handovers"]     / max(data["count"], 1) * 100, 2),
            "degradation_rate_pct": round(data["degradations"]  / max(data["count"], 1) * 100, 2),
        }
        for sc, data in sc_data.items()
    }


@app.get("/health", tags=["Health"])
async def health():
    return {"service": "dashboard", "status": "healthy", "timestamp": datetime.utcnow()}
