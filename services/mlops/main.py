"""
MLOps Service — Port 8005
Model training orchestration, versioning, and deployment registry.
"""
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from typing import List, Dict, Any, Optional
import uuid, time
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from shared.schemas import ModelVersion, TrainingRequest, ModelStatus, UserInfo, UserRole
from shared.config import settings
from services.auth.main import get_current_user, require_role

app = FastAPI(
    title="MLOps Service",
    description="Model training pipeline orchestration, versioning and deployment.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model Registry ────────────────────────────────────────────────────────────
MODEL_DIR = "/app/model_registry"

model_registry: Dict[str, List[ModelVersion]] = {
    "dso1": [ModelVersion(
        model_name="dso1", version="v1.0.0", status=ModelStatus.DEPLOYED,
        metrics={"accuracy": 0.884, "f1_score": 0.871, "roc_auc": 0.931},
        artifact_uri=MODEL_DIR + "/dso1/v1.0.0/model.pkl",
    )],
    "dso2": [ModelVersion(
        model_name="dso2", version="v1.0.0", status=ModelStatus.DEPLOYED,
        metrics={"mse": 4.21, "mae": 1.58, "rmse": 2.05},
        artifact_uri=MODEL_DIR + "/dso2/v1.0.0/model.pkl",
    )],
    "dso3": [ModelVersion(
        model_name="dso3", version="v1.0.0", status=ModelStatus.DEPLOYED,
        metrics={"silhouette_score": 0.62, "inertia": 1240.5},
        artifact_uri=MODEL_DIR + "/dso3/v1.0.0/model.pkl",
    )],
    "dso4": [ModelVersion(
        model_name="dso4", version="v1.0.0", status=ModelStatus.DEPLOYED,
        metrics={"accuracy": 0.901, "f1_score": 0.893, "roc_auc": 0.952},
        artifact_uri=MODEL_DIR + "/dso4/v1.0.0/model.pkl",
    )],
}

training_jobs: List[Dict[str, Any]] = []


# ── Training Simulation ────────────────────────────────────────────────────────

def simulate_training(job_id: str, request: TrainingRequest):
    start = time.time()
    print(f"[MLOps] Training job {job_id} started for {request.model_name}")
    time.sleep(2)

    if request.model_name in ("dso1", "dso4"):
        metrics = {
            "accuracy": round(0.88 + 0.02 * (hash(job_id) % 10) / 10, 4),
            "f1_score": round(0.87 + 0.02 * (hash(job_id) % 8)  / 10, 4),
            "roc_auc":  round(0.93 + 0.01 * (hash(job_id) % 5)  / 10, 4),
        }
    elif request.model_name == "dso2":
        metrics = {
            "mse":  round(4.0 + 0.5 * (hash(job_id) % 4) / 10, 4),
            "mae":  round(1.5 + 0.2 * (hash(job_id) % 5) / 10, 4),
            "rmse": round(2.0 + 0.3 * (hash(job_id) % 3) / 10, 4),
        }
    else:
        metrics = {"silhouette_score": round(0.60 + 0.05 * (hash(job_id) % 4) / 10, 4)}

    existing = model_registry.get(request.model_name, [])
    last_version = existing[-1].version if existing else "v0.0.0"
    major, minor, patch = map(int, last_version.lstrip("v").split("."))
    new_version = "v" + str(major) + "." + str(minor) + "." + str(patch + 1)

    new_model = ModelVersion(
        model_name=request.model_name,
        version=new_version,
        status=ModelStatus.TRAINED,
        metrics=metrics,
        artifact_uri=MODEL_DIR + "/" + request.model_name + "/" + new_version + "/model.pkl",
    )
    model_registry.setdefault(request.model_name, []).append(new_model)

    for job in training_jobs:
        if job["job_id"] == job_id:
            job.update({
                "status":       "completed",
                "completed_at": datetime.utcnow().isoformat(),
                "duration_s":   round(time.time() - start, 2),
                "new_version":  new_version,
                "metrics":      metrics,
            })
    print(f"[MLOps] Job {job_id} done → {new_version}")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/mlops/train", tags=["Training"])
async def trigger_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    current_user: UserInfo = Depends(require_role(UserRole.DATA_SCIENTIST, UserRole.ADMIN)),
):
    valid = {"dso1", "dso2", "dso3", "dso4"}
    if request.model_name not in valid:
        raise HTTPException(status_code=400, detail=f"model_name must be one of: {valid}")
    job_id = str(uuid.uuid4())
    training_jobs.append({
        "job_id":       job_id,
        "model_name":   request.model_name,
        "triggered_by": current_user.username,
        "dataset_path": request.dataset_path,
        "hyperparams":  request.hyperparams,
        "status":       "running",
        "started_at":   datetime.utcnow().isoformat(),
    })
    background_tasks.add_task(simulate_training, job_id, request)
    return {"job_id": job_id, "status": "running", "model_name": request.model_name}


@app.get("/mlops/jobs", tags=["Training"])
async def list_jobs(
    current_user: UserInfo = Depends(require_role(UserRole.DATA_SCIENTIST, UserRole.ADMIN)),
):
    return {"jobs": training_jobs}


@app.get("/mlops/jobs/{job_id}", tags=["Training"])
async def get_job(
    job_id: str,
    current_user: UserInfo = Depends(require_role(UserRole.DATA_SCIENTIST, UserRole.ADMIN)),
):
    for job in training_jobs:
        if job["job_id"] == job_id:
            return job
    raise HTTPException(status_code=404, detail=f"Job {job_id} not found")


@app.get("/mlops/registry", tags=["Model Registry"])
async def list_models(current_user: UserInfo = Depends(get_current_user)):
    return {
        model: [v.model_dump() for v in versions]
        for model, versions in model_registry.items()
    }


@app.get("/mlops/registry/{model_name}", tags=["Model Registry"])
async def get_model(
    model_name: str,
    current_user: UserInfo = Depends(get_current_user),
):
    versions = model_registry.get(model_name)
    if not versions:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    return {"model_name": model_name, "versions": [v.model_dump() for v in versions]}


@app.post("/mlops/registry/{model_name}/{version}/deploy", tags=["Model Registry"])
async def deploy_model(
    model_name: str,
    version: str,
    current_user: UserInfo = Depends(require_role(UserRole.ADMIN)),
):
    versions = model_registry.get(model_name, [])
    target = next((v for v in versions if v.version == version), None)
    if not target:
        raise HTTPException(status_code=404, detail=f"Version {version} not found")
    for v in versions:
        if v.status == ModelStatus.DEPLOYED and v.version != version:
            v.status = ModelStatus.RETIRED
    target.status = ModelStatus.DEPLOYED
    return {"deployed": True, "model_name": model_name, "version": version}


@app.post("/mlops/registry/{model_name}/{version}/retire", tags=["Model Registry"])
async def retire_model(
    model_name: str,
    version: str,
    current_user: UserInfo = Depends(require_role(UserRole.ADMIN)),
):
    versions = model_registry.get(model_name, [])
    target = next((v for v in versions if v.version == version), None)
    if not target:
        raise HTTPException(status_code=404, detail="Version not found")
    target.status = ModelStatus.RETIRED
    return {"retired": True, "model_name": model_name, "version": version}


@app.get("/health", tags=["Health"])
async def health():
    return {"service": "mlops", "status": "healthy", "timestamp": datetime.utcnow()}
