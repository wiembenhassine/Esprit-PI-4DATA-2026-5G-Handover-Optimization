"""
Shared Pydantic schemas across all microservices.
"""
from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime


class _BaseModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=())


# ─── Enums ────────────────────────────────────────────────────────────────────

class Scenario(str, Enum):
    HBAHN  = "hbahn"
    MOBILE = "mobile"
    STATIC = "static"

class UserRole(str, Enum):
    NETWORK_OPERATOR = "network_operator"
    DATA_SCIENTIST   = "data_scientist"
    ADMIN            = "admin"

class ModelStatus(str, Enum):
    TRAINED   = "trained"
    DEPLOYED  = "deployed"
    RETIRED   = "retired"
    TRAINING  = "training"

class DriftStatus(str, Enum):
    OK      = "ok"
    WARNING = "warning"
    DRIFT   = "drift"


# ─── Raw Input ─────────────────────────────────────────────────────────────────

class RawCellRecord(_BaseModel):
    """Raw 5G cell measurement record (from Data Sources)."""
    timestamp:       float         = Field(..., description="Unix timestamp (seconds)")
    scenario:        Scenario
    physical_cellid: int
    rsrp:            float         = Field(..., ge=-140, le=-44, description="dBm")
    rsrq:            float         = Field(..., ge=-20,  le=-3,  description="dB")
    sinr:            float         = Field(..., ge=-20,  le=30,  description="dB")
    cqi:             Optional[int] = Field(None, ge=0, le=15)
    tx_power:        Optional[float] = None
    ta:              Optional[float] = None
    ss_rsrp:         Optional[float] = None
    ss_sinr:         Optional[float] = None
    lte_mcs:         Optional[int]   = None
    lte_ri:          Optional[int]   = None
    earfcn:          Optional[int]   = None
    velocity:        Optional[float] = None
    latitude:        Optional[float] = None
    longitude:       Optional[float] = None


class RawNeighborRecord(_BaseModel):
    """Neighboring cell measurements."""
    timestamp:           float
    scenario:            Scenario
    neighbor_cellid:     int
    neighbor_rsrp:       float
    neighbor_rsrq:       Optional[float] = None
    neighbor_sinr:       Optional[float] = None


class RawIngestionRequest(_BaseModel):
    """Batch ingestion payload."""
    cell_records:     List[RawCellRecord]
    neighbor_records: Optional[List[RawNeighborRecord]] = []
    source_id:        str = Field(..., description="Data source identifier")


# ─── Processed / Engineered Features ──────────────────────────────────────────

class EngineeredFeatures(_BaseModel):
    """Feature vector after processing — input to DSO models."""
    timestamp:        float
    scenario:         Scenario
    # Core RF
    rsrp:             float
    rsrq:             float
    sinr:             float
    cqi:              Optional[float] = None
    tx_power:         Optional[float] = None
    ta:               Optional[float] = None
    ss_rsrp:          Optional[float] = None
    ss_sinr:          Optional[float] = None
    lte_mcs:          Optional[float] = None
    lte_ri:           Optional[float] = None
    earfcn:           Optional[float] = None
    velocity:         Optional[float] = None
    # Lag / trend features (engineered)
    rsrp_lag1:        Optional[float] = None
    rsrp_lag3:        Optional[float] = None
    rsrp_lag5:        Optional[float] = None
    rsrp_slope3:      Optional[float] = None
    rsrp_slope5:      Optional[float] = None
    sinr_lag1:        Optional[float] = None
    sinr_slope3:      Optional[float] = None
    # Neighbor aggregates
    neighbor_gap:     Optional[float] = None  # best_neighbor_rsrp - serving_rsrp
    best_neighbor_rsrp: Optional[float] = None
    n_neighbors:      Optional[int]   = None
    # Cell load
    cell_load_proxy:  Optional[float] = None
    # Temporal
    hour_of_day:      Optional[int]   = None
    day_of_week:      Optional[int]   = None
    time_bin:         Optional[int]   = None
    # Imputation flag
    latency_is_imputed: Optional[int] = None
    # Cell load / history features (from notebook Phase 2 cell-load merge)
    cell_hist_datarate_mean: Optional[float] = None   # mean historical datarate for serving cell
    cell_load_drop_flag:     Optional[int]   = None   # 1 if cell load caused datarate drop


# ─── DSO Outputs ──────────────────────────────────────────────────────────────

class DSO1Output(_BaseModel):
    """Signal Degradation Prediction (Binary Classification)."""
    is_degrading:        bool
    degradation_prob:    float  = Field(..., ge=0, le=1)
    rsrp_future_5_pred:  float
    shap_values:         Optional[Dict[str, float]] = None
    model_version:       str = "dso1_v1"


class DSO2Output(_BaseModel):
    """Handover Target Estimation (Dual Regression)."""
    predicted_neighbor_gap:   float   # signal gain if handed over
    predicted_best_rsrp:      float   # absolute quality of best neighbour
    recommended_cellid:       Optional[int] = None
    model_version:            str = "dso2_v1"


class DSO3Output(_BaseModel):
    """User State Profiling (Clustering)."""
    cluster_id:    int
    cluster_label: str   # e.g. "good", "fair", "cell_edge", "congested"
    cluster_probs: Optional[Dict[int, float]] = None
    model_version: str = "dso3_v1"


class DSO4Output(_BaseModel):
    """Master Handover Controller (Binary Classification)."""
    handover_recommended: bool
    handover_prob:        float = Field(..., ge=0, le=1)
    confidence:           str   # "high" | "medium" | "low"
    model_version:        str = "dso4_v1"


class PredictionResult(_BaseModel):
    """Chained DSO pipeline result."""
    request_id:  str
    timestamp:   float
    scenario:    Scenario
    dso1:        DSO1Output
    dso2:        DSO2Output
    dso3:        DSO3Output
    dso4:        DSO4Output
    latency_ms:  float
    created_at:  datetime = Field(default_factory=datetime.utcnow)


# ─── Monitoring ────────────────────────────────────────────────────────────────

class ModelMetrics(_BaseModel):
    model_name:   str
    accuracy:     Optional[float] = None
    f1_score:     Optional[float] = None
    roc_auc:      Optional[float] = None
    mse:          Optional[float] = None
    mae:          Optional[float] = None
    n_predictions: int
    drift_status: DriftStatus
    drift_score:  float
    evaluated_at: datetime = Field(default_factory=datetime.utcnow)


class Alert(_BaseModel):
    alert_id:    str
    severity:    str   # "info" | "warning" | "critical"
    model_name:  str
    message:     str
    metric:      str
    value:       float
    threshold:   float
    created_at:  datetime = Field(default_factory=datetime.utcnow)


# ─── Auth ──────────────────────────────────────────────────────────────────────

class TokenRequest(_BaseModel):
    username: str
    password: str

class TokenResponse(_BaseModel):
    access_token:  str
    token_type:    str = "bearer"
    expires_in:    int
    role:          UserRole

class UserInfo(_BaseModel):
    user_id:  str
    username: str
    role:     UserRole
    email:    Optional[str] = None


# ─── MLOps ────────────────────────────────────────────────────────────────────

class ModelVersion(_BaseModel):
    model_name:   str
    version:      str
    status:       ModelStatus
    metrics:      Dict[str, float]
    artifact_uri: str
    created_at:   datetime = Field(default_factory=datetime.utcnow)

class TrainingRequest(_BaseModel):
    model_name:     str   # "dso1" | "dso2" | "dso3" | "dso4"
    dataset_path:   str
    hyperparams:    Optional[Dict[str, Any]] = {}
    triggered_by:   str = "manual"
