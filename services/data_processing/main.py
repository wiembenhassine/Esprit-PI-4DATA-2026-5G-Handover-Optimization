"""
Data Processing Service — Port 8002
Cleans raw records, engineers features, and produces EngineeredFeatures
ready for consumption by the Prediction Service.

Feature engineering mirrors the notebook (Phase 2):
  - Imputation (3-layer strategy)
  - Lag features: rsrp_lag1/3/5, sinr_lag1
  - Slope features: rsrp_slope3/5, sinr_slope3
  - Neighbor aggregation: neighbor_gap, best_neighbor_rsrp
  - Temporal features: hour_of_day, day_of_week, time_bin
  - Target engineering (training only): target_is_degrading, target_ho_flag
"""
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from shared.schemas import (
    RawCellRecord, RawNeighborRecord, EngineeredFeatures,
    Scenario, UserInfo, UserRole
)
from shared.config import settings
from services.auth.main import get_current_user, require_role

app = FastAPI(
    title="Data Processing Service",
    description="Cleans, transforms, and engineers features for model consumption.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Feature Engineering Logic ─────────────────────────────────────────────────

class FeatureEngineer:
    """
    Replicates the feature engineering pipeline from the 5G Handover notebook.

    Key engineering steps:
      1. Imputation (median for RF features, 0 for mobility)
      2. Lag features (1, 3, 5 steps)
      3. Slope features (rolling linear regression proxy)
      4. Neighbor aggregation
      5. Temporal encoding
    """

    CORE_RF_FEATURES = [
        "rsrp", "rsrq", "sinr", "cqi", "tx_power", "ta",
        "ss_rsrp", "ss_sinr", "lte_mcs", "lte_ri", "earfcn",
    ]

    # Medians derived from notebook EDA (approximate; replace with fitted values)
    RF_MEDIANS: Dict[str, float] = {
        "rsrp": -88.0, "rsrq": -10.5, "sinr": 12.0, "cqi": 10,
        "tx_power": 15.0, "ta": 3.0, "ss_rsrp": -85.0, "ss_sinr": 14.0,
        "lte_mcs": 18, "lte_ri": 2, "earfcn": 1300,
    }

    def _impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """3-layer imputation: forward-fill → backward-fill → median."""
        for col in self.CORE_RF_FEATURES:
            if col in df.columns:
                df[col] = (df[col].ffill().bfill()
                           .fillna(self.RF_MEDIANS.get(col, 0)))
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(self.RF_MEDIANS.get(col, 0))
        if "velocity" in df.columns:
            df["velocity"] = pd.to_numeric(df["velocity"], errors="coerce").fillna(0.0)
        return df

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rolling lag features as in notebook Phase 2."""
        for lag in settings.LAG_STEPS:
            df[f"rsrp_lag{lag}"] = df["rsrp"].shift(lag)
        df["sinr_lag1"] = df["sinr"].shift(1)
        return df

    def _add_slope_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Slope = (current - lag_N) / N as a cheap linear-regression proxy.
        Matches notebook: rsrp_slope3, rsrp_slope5, sinr_slope3
        """
        df["rsrp_slope3"] = (df["rsrp"] - df["rsrp_lag3"]) / 3
        df["rsrp_slope5"] = (df["rsrp"] - df["rsrp_lag5"]) / 5
        df["sinr_slope3"] = (df["sinr"] - df["sinr_lag1"].shift(2)) / 3
        return df

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Hour, day-of-week, and time_bin (6 bins of 4 hours)."""
        ts = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        df["hour_of_day"] = ts.dt.hour
        df["day_of_week"] = ts.dt.dayofweek
        df["time_bin"]    = ts.dt.hour // 4  # 0-5
        return df

    def _add_neighbor_features(
        self,
        df: pd.DataFrame,
        neighbor_df: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """Aggregate neighbor cells per timestamp."""
        if neighbor_df is None or neighbor_df.empty:
            df["neighbor_gap"]       = np.nan
            df["best_neighbor_rsrp"] = np.nan
            df["n_neighbors"]        = 0
            return df

        agg = (
            neighbor_df
            .groupby("timestamp")["neighbor_rsrp"]
            .agg(best_neighbor_rsrp="max", n_neighbors="count")
            .reset_index()
        )
        df = df.merge(agg, on="timestamp", how="left")
        df["neighbor_gap"] = df["best_neighbor_rsrp"] - df["rsrp"]
        df["n_neighbors"]  = df["n_neighbors"].fillna(0).astype(int)
        return df

    def _add_target_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        DSO targets — only used during training data preparation.
        DSO1: target_is_degrading = 1 if 5-step-ahead RSRP slope < -3 dBm
        DSO4: target_ho_flag = 1 (placeholder; real value from datarate join)
        """
        df["rsrp_future_5"] = df["rsrp"].shift(-5)
        df["rsrp_future_slope"] = (df["rsrp_future_5"] - df["rsrp"]) / 5
        df["target_is_degrading"] = (
            df["rsrp_future_slope"] < settings.DEGRADATION_THRESHOLD
        ).astype(int)
        # DSO4 target requires datarate — set as NaN here, populated by merge
        df["target_ho_flag"] = np.nan
        return df

    def process(
        self,
        cell_records: List[RawCellRecord],
        neighbor_records: Optional[List[RawNeighborRecord]] = None,
        include_targets: bool = False,
    ) -> List[EngineeredFeatures]:
        """Full processing pipeline → list of EngineeredFeatures."""
        if not cell_records:
            return []

        df = pd.DataFrame([r.model_dump() for r in cell_records])
        df = df.sort_values("timestamp").reset_index(drop=True)
        df = self._impute(df)
        df = self._add_lag_features(df)
        df = self._add_slope_features(df)
        df = self._add_temporal_features(df)

        neighbor_df = None
        if neighbor_records:
            neighbor_df = pd.DataFrame([r.model_dump() for r in neighbor_records])
        df = self._add_neighbor_features(df, neighbor_df)

        if include_targets:
            df = self._add_target_features(df)

        # Imputation flag for latency
        df["latency_is_imputed"] = 1  # assume imputed unless latency is joined

        # Fill lag NaNs (first few rows won't have history)
        lag_cols = [c for c in df.columns if "lag" in c or "slope" in c]
        df[lag_cols] = df[lag_cols].bfill().fillna(0)

        # Convert to output schema
        features = []
        for _, row in df.iterrows():
            def safe(val):
                """Convert NaN/inf to None for JSON safety."""
                import math
                if val is None:
                    return None
                try:
                    f = float(val)
                    return None if (math.isnan(f) or math.isinf(f)) else f
                except (TypeError, ValueError):
                    return val
            try:
                features.append(EngineeredFeatures(
                    timestamp=row["timestamp"],
                    scenario=row["scenario"],
                    rsrp=row["rsrp"], rsrq=row["rsrq"], sinr=row["sinr"],
                    cqi=safe(row.get("cqi")), tx_power=safe(row.get("tx_power")),
                    ta=safe(row.get("ta")), ss_rsrp=safe(row.get("ss_rsrp")),
                    ss_sinr=safe(row.get("ss_sinr")), lte_mcs=safe(row.get("lte_mcs")),
                    lte_ri=safe(row.get("lte_ri")), earfcn=safe(row.get("earfcn")),
                    velocity=safe(row.get("velocity")),
                    rsrp_lag1=safe(row.get("rsrp_lag1")), rsrp_lag3=safe(row.get("rsrp_lag3")),
                    rsrp_lag5=safe(row.get("rsrp_lag5")), rsrp_slope3=safe(row.get("rsrp_slope3")),
                    rsrp_slope5=safe(row.get("rsrp_slope5")), sinr_lag1=safe(row.get("sinr_lag1")),
                    sinr_slope3=safe(row.get("sinr_slope3")),
                    neighbor_gap=safe(row.get("neighbor_gap")),
                    best_neighbor_rsrp=safe(row.get("best_neighbor_rsrp")),
                    n_neighbors=int(row.get("n_neighbors", 0)),
                    hour_of_day=int(row.get("hour_of_day", 0)),
                    day_of_week=int(row.get("day_of_week", 0)),
                    time_bin=int(row.get("time_bin", 0)),
                    latency_is_imputed=int(row.get("latency_is_imputed", 1)),
                ))
            except Exception as e:
                print(f"[Processing] Row conversion error: {e}")
        return features


engineer = FeatureEngineer()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/process/batch", response_model=List[EngineeredFeatures], tags=["Processing"])
async def process_batch(
    payload: Dict[str, Any],
    current_user: UserInfo = Depends(get_current_user),
):
    """
    Process a raw ingestion batch into engineered features.

    Called internally by the Data Ingestion Service.
    Returns a list of EngineeredFeatures ready for the Prediction Service.
    """
    try:
        cell_records = [RawCellRecord(**r) for r in payload.get("cell_records", [])]
        neighbor_records = [RawNeighborRecord(**r) for r in payload.get("neighbor_records", [])]
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Payload parse error: {e}")

    features = engineer.process(cell_records, neighbor_records)
    return features


@app.post("/process/single", response_model=EngineeredFeatures, tags=["Processing"])
async def process_single(
    record: RawCellRecord,
    current_user: UserInfo = Depends(get_current_user),
):
    """Process a single cell record (useful for real-time / testing)."""
    features = engineer.process([record])
    if not features:
        raise HTTPException(status_code=422, detail="Feature engineering produced no output")
    return features[0]


@app.post("/process/training-dataset", tags=["Processing"])
async def build_training_dataset(
    payload: Dict[str, Any],
    current_user: UserInfo = Depends(require_role(UserRole.DATA_SCIENTIST, UserRole.ADMIN)),
):
    """
    Build a training dataset (includes DSO targets).
    Data Scientist / Admin only.
    """
    cell_records    = [RawCellRecord(**r) for r in payload.get("cell_records", [])]
    neighbor_records = [RawNeighborRecord(**r) for r in payload.get("neighbor_records", [])]
    features = engineer.process(cell_records, neighbor_records, include_targets=True)
    return {
        "n_samples": len(features),
        "features":  [f.model_dump() for f in features],
        "columns":   list(features[0].model_dump().keys()) if features else [],
    }


@app.get("/process/feature-info", tags=["Processing"])
async def feature_info(current_user: UserInfo = Depends(get_current_user)):
    """Describe the feature engineering pipeline."""
    return {
        "core_rf_features":   FeatureEngineer.CORE_RF_FEATURES,
        "lag_steps":          settings.LAG_STEPS,
        "slope_window":       settings.SLOPE_WINDOW,
        "degradation_threshold_dbm": settings.DEGRADATION_THRESHOLD,
        "imputation_strategy": "forward-fill → backward-fill → median",
        "temporal_features":  ["hour_of_day", "day_of_week", "time_bin"],
        "neighbor_aggregates": ["neighbor_gap", "best_neighbor_rsrp", "n_neighbors"],
    }


@app.get("/health", tags=["Health"])
async def health():
    return {"service": "data-processing", "status": "healthy", "timestamp": datetime.utcnow()}
