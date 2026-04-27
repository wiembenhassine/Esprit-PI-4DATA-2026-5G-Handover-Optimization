"""
API Gateway — Port 8000
Single entry point for all external consumers (UI, External Applications).
Routes requests to internal microservices, enforces authentication,
and provides a unified Swagger UI at /docs.

Exposes only prediction results and insights — never raw data.
"""
from fastapi import FastAPI, Depends, HTTPException, Request, UploadFile, File
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
from typing import List, Dict, Any, Optional
import httpx
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from shared.schemas import (
    RawIngestionRequest, EngineeredFeatures,
    TrainingRequest, UserInfo, UserRole
)
from shared.config import settings
from services.auth.main import get_current_user, require_role

oauth2_scheme_gw = OAuth2PasswordBearer(tokenUrl="/auth/token", auto_error=False)

app = FastAPI(
    title="AI Handover Optimization Platform — API Gateway",
    description="""
## AI Handover Optimization Platform

Unified API Gateway for the 5G Handover Optimization Platform.

### Architecture
- **Data Ingestion Service** — Accepts raw network measurements
- **Data Processing Service** — Feature engineering pipeline
- **Prediction Service** — Chained DSO1→DSO2→DSO3→DSO4 models
- **Monitoring Service** — Drift detection and alerting
- **Dashboard Service** — Operator views and SHAP explanations
- **MLOps Service** — Training, versioning, and deployment
- **Auth Service** — JWT authentication and RBAC

### Roles
| Role | Permissions |
|------|-------------|
| `network_operator` | Read predictions, dashboard, alerts |
| `data_scientist` | All operator + training, model evaluation |
| `admin` | All + user management, model deployment |

### Authentication
Use `/auth/token` with username/password to receive a JWT bearer token.
Include it as `Authorization: Bearer <token>` in all subsequent requests.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── HTTP Client factory ───────────────────────────────────────────────────────

def make_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=30.0)

def _svc_headers(user_token: str = None) -> dict:
    """
    For calls that proxy a user request: forward the real user token.
    For internal aggregation calls: use the service token.
    """
    token = user_token or settings.INTERNAL_SERVICE_TOKEN
    return {"Authorization": f"Bearer {token}"}

async def proxy(client: httpx.AsyncClient, method: str, url: str, token: str = None, **kwargs):
    try:
        headers = kwargs.pop("headers", {})
        headers.update(_svc_headers(token))
        r = await client.request(method, url, headers=headers, **kwargs)
        return r.json()
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail=f"Upstream service unavailable: {url}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gateway error: {str(e)}")


# ══════════════════════════════════════════════════════════════════════════════
# AUTH ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/auth/token", tags=["🔐 Authentication"],
          summary="Login and receive JWT token")
async def gateway_login(request: Request):
    """Proxy to Auth Service. Use this to obtain a JWT bearer token."""
    form = await request.form()
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(
                f"{settings.AUTH_URL}/auth/token",
                data=dict(form),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
        return JSONResponse(content=r.json(), status_code=r.status_code)
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Auth service unavailable")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Login error: {str(e)}")


@app.get("/auth/me", tags=["🔐 Authentication"])
async def gateway_me(
    current_user: UserInfo = Depends(get_current_user),
    raw_token: str = Depends(oauth2_scheme_gw),
):
    return current_user


# ══════════════════════════════════════════════════════════════════════════════
# INGESTION ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/ingest/batch", tags=["📡 Data Ingestion"],
          summary="Ingest raw cell measurement batch")
async def gateway_ingest_batch(
    request: RawIngestionRequest,
    current_user: UserInfo = Depends(get_current_user),
    raw_token: str = Depends(oauth2_scheme_gw),
):
    """
    Submit a batch of raw 5G cell measurements for processing and prediction.
    Data is validated, processed through the feature engineering pipeline,
    and fed to the DSO prediction chain.
    """
    async with make_client() as client:
        return await proxy(client, "POST",
            f"{settings.DATA_INGESTION_URL}/ingest/batch",
            json=request.model_dump(),
        )


@app.get("/ingest/stats", tags=["📡 Data Ingestion"])
async def gateway_ingest_stats(
    current_user: UserInfo = Depends(require_role(UserRole.DATA_SCIENTIST, UserRole.ADMIN)),
    raw_token: str = Depends(oauth2_scheme_gw),
):
    async with make_client() as client:
        return await proxy(client, "GET", f"{settings.DATA_INGESTION_URL}/ingest/stats", token=raw_token)


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION ROUTES  (external apps receive results only — not raw data)
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/predict", tags=["🤖 Predictions"],
          summary="Run DSO1→DSO2→DSO3→DSO4 prediction pipeline")
async def gateway_predict(
    features: EngineeredFeatures,
    current_user: UserInfo = Depends(get_current_user),
    raw_token: str = Depends(oauth2_scheme_gw),
):
    """
    Submit engineered features and receive the full chained DSO prediction.

    Returns:
    - **DSO1**: Signal degradation probability + SHAP explanation
    - **DSO2**: Best neighbor RSRP prediction + handover target gain
    - **DSO3**: Network state cluster (good/fair/cell_edge/congested)
    - **DSO4**: Final handover recommendation + confidence
    """
    async with make_client() as client:
        return await proxy(client, "POST",
            f"{settings.PREDICTION_URL}/predict",
            json=features.model_dump(),
        )


@app.post("/predict/batch", tags=["🤖 Predictions"])
async def gateway_predict_batch(
    feature_list: List[EngineeredFeatures],
    current_user: UserInfo = Depends(get_current_user),
    raw_token: str = Depends(oauth2_scheme_gw),
):
    """Batch prediction (up to 1000 records per request)."""
    async with make_client() as client:
        return await proxy(client, "POST",
            f"{settings.PREDICTION_URL}/predict/batch",
            json=[f.model_dump() for f in feature_list],
        )


@app.get("/predict/history", tags=["🤖 Predictions"])
async def gateway_prediction_history(
    limit: int = 50,
    scenario: Optional[str] = None,
    current_user: UserInfo = Depends(get_current_user),
    raw_token: str = Depends(oauth2_scheme_gw),
):
    url = f"{settings.PREDICTION_URL}/predict/history?limit={limit}"
    if scenario:
        url += f"&scenario={scenario}"
    async with make_client() as client:
        return await proxy(client, "GET", url, token=raw_token)


@app.get("/predict/stats", tags=["🤖 Predictions"])
async def gateway_predict_stats(current_user: UserInfo = Depends(get_current_user), raw_token: str = Depends(oauth2_scheme_gw)):
    async with make_client() as client:
        return await proxy(client, "GET", f"{settings.PREDICTION_URL}/predict/stats", token=raw_token)


@app.get("/predict/models", tags=["🤖 Predictions"])
async def gateway_predict_models(current_user: UserInfo = Depends(get_current_user), raw_token: str = Depends(oauth2_scheme_gw)):
    async with make_client() as client:
        return await proxy(client, "GET", f"{settings.PREDICTION_URL}/predict/models", token=raw_token)


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/dashboard/overview", tags=["📊 Dashboard"])
async def gateway_overview(current_user: UserInfo = Depends(get_current_user), raw_token: str = Depends(oauth2_scheme_gw)):
    """Platform overview combining stats from all services."""
    async with make_client() as client:
        return await proxy(client, "GET", f"{settings.DASHBOARD_URL}/dashboard/overview", token=raw_token)


@app.get("/dashboard/live-feed", tags=["📊 Dashboard"])
async def gateway_live_feed(
    limit: int = 20,
    current_user: UserInfo = Depends(get_current_user),
    raw_token: str = Depends(oauth2_scheme_gw),
):
    async with make_client() as client:
        return await proxy(client, "GET",
            f"{settings.DASHBOARD_URL}/dashboard/live-feed?limit={limit}")


@app.get("/dashboard/kpis", tags=["📊 Dashboard"])
async def gateway_kpis(current_user: UserInfo = Depends(get_current_user), raw_token: str = Depends(oauth2_scheme_gw)):
    async with make_client() as client:
        return await proxy(client, "GET", f"{settings.DASHBOARD_URL}/dashboard/kpis", token=raw_token)


@app.get("/dashboard/shap/{request_id}", tags=["📊 Dashboard"])
async def gateway_shap(request_id: str, current_user: UserInfo = Depends(get_current_user)):
    async with make_client() as client:
        return await proxy(client, "GET",
            f"{settings.DASHBOARD_URL}/dashboard/shap/{request_id}")


@app.get("/dashboard/scenarios", tags=["📊 Dashboard"])
async def gateway_scenarios(current_user: UserInfo = Depends(get_current_user),raw_token: str = Depends(oauth2_scheme_gw)):
    async with make_client() as client:
        return await proxy(client, "GET", f"{settings.DASHBOARD_URL}/dashboard/scenarios", token=raw_token)


# ══════════════════════════════════════════════════════════════════════════════
# MONITORING ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/monitor/drift", tags=["📈 Monitoring"])
async def gateway_drift(current_user: UserInfo = Depends(get_current_user), raw_token: str = Depends(oauth2_scheme_gw)):
    """Check concept drift scores (PSI) for all deployed models."""
    async with make_client() as client:
        return await proxy(client, "GET", f"{settings.MONITORING_URL}/monitor/drift", token=raw_token)


@app.get("/monitor/metrics", tags=["📈 Monitoring"])
async def gateway_metrics(
    model_name: Optional[str] = None,
    current_user: UserInfo = Depends(get_current_user),
    raw_token: str = Depends(oauth2_scheme_gw),
):
    url = f"{settings.MONITORING_URL}/monitor/metrics"
    if model_name:
        url += f"?model_name={model_name}"
    async with make_client() as client:
        return await proxy(client, "GET", url, token=raw_token)


@app.get("/monitor/alerts", tags=["📈 Monitoring"])
async def gateway_alerts(
    severity: Optional[str] = None,
    limit: int = 50,
    current_user: UserInfo = Depends(get_current_user),
    raw_token: str = Depends(oauth2_scheme_gw),
):
    url = f"{settings.MONITORING_URL}/monitor/alerts?limit={limit}"
    if severity:
        url += f"&severity={severity}"
    async with make_client() as client:
        return await proxy(client, "GET", url, token=raw_token)


@app.post("/monitor/evaluate", tags=["📈 Monitoring"])
async def gateway_evaluate(
    payload: Dict[str, Any],
    current_user: UserInfo = Depends(require_role(UserRole.DATA_SCIENTIST, UserRole.ADMIN)),
    raw_token: str = Depends(oauth2_scheme_gw),
):
    async with make_client() as client:
        return await proxy(client, "POST",
            f"{settings.MONITORING_URL}/monitor/evaluate", json=payload)


# ══════════════════════════════════════════════════════════════════════════════
# MLOPS ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/mlops/train", tags=["⚙️ MLOps"])
async def gateway_train(
    request: TrainingRequest,
    current_user: UserInfo = Depends(require_role(UserRole.DATA_SCIENTIST, UserRole.ADMIN)),
    raw_token: str = Depends(oauth2_scheme_gw),
):
    """Trigger model training. Data Scientist / Admin only."""
    async with make_client() as client:
        return await proxy(client, "POST",
            f"{settings.MLOPS_URL}/mlops/train", json=request.model_dump())


@app.get("/mlops/jobs", tags=["⚙️ MLOps"])
async def gateway_jobs(
    current_user: UserInfo = Depends(require_role(UserRole.DATA_SCIENTIST, UserRole.ADMIN)),
    raw_token: str = Depends(oauth2_scheme_gw),
):
    async with make_client() as client:
        return await proxy(client, "GET", f"{settings.MLOPS_URL}/mlops/jobs", token=raw_token)


@app.get("/mlops/registry", tags=["⚙️ MLOps"])
async def gateway_registry(current_user: UserInfo = Depends(get_current_user), raw_token: str = Depends(oauth2_scheme_gw)):
    async with make_client() as client:
        return await proxy(client, "GET", f"{settings.MLOPS_URL}/mlops/registry", token=raw_token)


@app.post("/mlops/registry/{model_name}/{version}/deploy", tags=["⚙️ MLOps"])
async def gateway_deploy(
    model_name: str,
    version: str,
    current_user: UserInfo = Depends(require_role(UserRole.ADMIN)),
    raw_token: str = Depends(oauth2_scheme_gw),
):
    """Promote a model version to deployed. Admin only."""
    async with make_client() as client:
        return await proxy(client, "POST",
            f"{settings.MLOPS_URL}/mlops/registry/{model_name}/{version}/deploy")


# ══════════════════════════════════════════════════════════════════════════════
# PLATFORM HEALTH
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health", tags=["🏥 Health"], summary="Platform-wide health check")
async def platform_health():
    """Check health of all microservices."""
    services = {
        "auth":            settings.AUTH_URL,
        "data_ingestion":  settings.DATA_INGESTION_URL,
        "data_processing": settings.DATA_PROCESSING_URL,
        "prediction":      settings.PREDICTION_URL,
        "monitoring":      settings.MONITORING_URL,
        "dashboard":       settings.DASHBOARD_URL,
        "mlops":           settings.MLOPS_URL,
    }
    results = {}
    async with make_client() as client:
        for name, base_url in services.items():
            try:
                r = await client.get(f"{base_url}/health", timeout=3.0)
                results[name] = r.json()
            except Exception:
                results[name] = {"status": "unreachable"}

    overall = "healthy" if all(
        v.get("status") == "healthy" for v in results.values()
    ) else "degraded"

    return {
        "platform_status": overall,
        "checked_at":      datetime.utcnow().isoformat(),
        "services":        results,
    }
