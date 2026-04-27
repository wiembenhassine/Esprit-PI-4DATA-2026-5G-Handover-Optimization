"""
Data Ingestion Service — Port 8001
Receives raw data from cellular network and GPS/mobility sources.
Validates, queues, and forwards to the Data Processing Service.
"""
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from typing import List, Dict, Any
import httpx
import uuid
import io
import csv
import json
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from shared.schemas import (
    RawIngestionRequest, RawCellRecord, RawNeighborRecord,
    UserInfo, UserRole
)
from shared.config import settings
from services.auth.main import get_current_user, require_role

app = FastAPI(
    title="Data Ingestion Service",
    description="Receives raw cellular and GPS data, validates it, and forwards to processing.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory ingestion log (replace with message queue like Kafka/RabbitMQ) ──
ingestion_log: List[Dict[str, Any]] = []
QUEUE: List[RawIngestionRequest] = []


# ── Background forwarding ─────────────────────────────────────────────────────

async def forward_to_processing(request: RawIngestionRequest, batch_id: str):
    """Send validated raw data to the Data Processing Service."""
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            payload = request.model_dump()
            payload["batch_id"] = batch_id
            resp = await client.post(
                f"{settings.DATA_PROCESSING_URL}/process/batch",
                json=payload,
            )
            status_code = resp.status_code
        except Exception as e:
            status_code = -1
            print(f"[Ingestion] Forward failed: {e}")

    ingestion_log.append({
        "batch_id": batch_id,
        "n_cell_records": len(request.cell_records),
        "n_neighbor_records": len(request.neighbor_records),
        "source_id": request.source_id,
        "forwarded_status": status_code,
        "ingested_at": datetime.utcnow().isoformat(),
    })


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/ingest/batch", tags=["Ingestion"])
async def ingest_batch(
    request: RawIngestionRequest,
    background_tasks: BackgroundTasks,
    current_user: UserInfo = Depends(get_current_user),
):
    """
    Ingest a batch of raw cell + neighbor measurements.

    Accepts JSON payload with cell_records and optional neighbor_records.
    Data is validated then forwarded asynchronously to the Data Processing Service.
    """
    batch_id = str(uuid.uuid4())

    # Basic validation
    if not request.cell_records:
        raise HTTPException(status_code=422, detail="cell_records cannot be empty")

    # Scenario consistency check
    scenarios = {r.scenario for r in request.cell_records}
    if len(scenarios) > 1:
        raise HTTPException(
            status_code=422,
            detail=f"Mixed scenarios in one batch: {scenarios}. Split by scenario.",
        )

    # Queue async forwarding
    background_tasks.add_task(forward_to_processing, request, batch_id)

    return {
        "batch_id": batch_id,
        "accepted_cell_records": len(request.cell_records),
        "accepted_neighbor_records": len(request.neighbor_records),
        "scenario": list(scenarios)[0],
        "source_id": request.source_id,
        "status": "queued",
        "message": "Data accepted and queued for processing.",
    }


@app.post("/ingest/csv", tags=["Ingestion"])
async def ingest_csv(
    file: UploadFile = File(...),
    scenario: str = "hbahn",
    source_id: str = "csv_upload",
    background_tasks: BackgroundTasks = BackgroundTasks(),
    current_user: UserInfo = Depends(require_role(UserRole.DATA_SCIENTIST, UserRole.ADMIN)),
):
    """
    Upload a CSV file for ingestion (Data Scientist / Admin only).

    Columns expected: timestamp, physical_cellid, rsrp, rsrq, sinr
    Optional: cqi, tx_power, ta, ss_rsrp, ss_sinr, lte_mcs, lte_ri, earfcn, velocity
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    content = await file.read()
    reader = csv.DictReader(io.StringIO(content.decode("utf-8")))

    records = []
    errors = []
    for i, row in enumerate(reader):
        try:
            records.append(RawCellRecord(
                timestamp=float(row["timestamp"]),
                scenario=scenario,
                physical_cellid=int(row["physical_cellid"]),
                rsrp=float(row["rsrp"]),
                rsrq=float(row["rsrq"]),
                sinr=float(row["sinr"]),
                cqi=int(row["cqi"]) if row.get("cqi") else None,
                tx_power=float(row["tx_power"]) if row.get("tx_power") else None,
                ta=float(row["ta"]) if row.get("ta") else None,
                velocity=float(row["velocity"]) if row.get("velocity") else None,
            ))
        except Exception as e:
            errors.append({"row": i + 2, "error": str(e)})

    if not records:
        raise HTTPException(status_code=422, detail=f"No valid records parsed. Errors: {errors[:5]}")

    batch_id = str(uuid.uuid4())
    req = RawIngestionRequest(cell_records=records, source_id=source_id)
    background_tasks.add_task(forward_to_processing, req, batch_id)

    return {
        "batch_id": batch_id,
        "parsed_records": len(records),
        "parse_errors": len(errors),
        "sample_errors": errors[:3],
        "status": "queued",
    }


@app.get("/ingest/log", tags=["Ingestion"])
async def get_ingestion_log(
    limit: int = 50,
    current_user: UserInfo = Depends(require_role(UserRole.DATA_SCIENTIST, UserRole.ADMIN)),
):
    """Retrieve recent ingestion history."""
    return {
        "total": len(ingestion_log),
        "items": ingestion_log[-limit:],
    }


@app.get("/ingest/stats", tags=["Ingestion"])
async def get_ingestion_stats(current_user: UserInfo = Depends(get_current_user)):
    """Summary statistics of ingested data."""
    total_cell = sum(e.get("n_cell_records", 0) for e in ingestion_log)
    total_neighbor = sum(e.get("n_neighbor_records", 0) for e in ingestion_log)
    successful = sum(1 for e in ingestion_log if e.get("forwarded_status") == 200)
    return {
        "total_batches": len(ingestion_log),
        "total_cell_records": total_cell,
        "total_neighbor_records": total_neighbor,
        "successful_forwards": successful,
        "failed_forwards": len(ingestion_log) - successful,
    }


@app.get("/health", tags=["Health"])
async def health():
    return {"service": "data-ingestion", "status": "healthy", "timestamp": datetime.utcnow()}
