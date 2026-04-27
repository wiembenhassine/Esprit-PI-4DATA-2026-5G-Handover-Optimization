"""
Shared configuration — loaded from environment variables with sane defaults.
"""
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # ── JWT ───────────────────────────────────────────────────────────────────
    SECRET_KEY:            str  = "CHANGE_ME_IN_PROD_use_a_32char_random_string"
    ALGORITHM:             str  = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # ── Internal service-to-service token ─────────────────────────────────────
    # All services use this when calling each other (avoids 401 on inter-service calls)
    INTERNAL_SERVICE_TOKEN: str = "internal-service-secret-token-change-in-prod"

    # ── Service URLs (for inter-service calls) ─────────────────────────────────
    DATA_INGESTION_URL:    str  = "http://data-ingestion:8001"
    DATA_PROCESSING_URL:   str  = "http://data-processing:8002"
    PREDICTION_URL:        str  = "http://prediction:8003"
    MONITORING_URL:        str  = "http://monitoring:8004"
    MLOPS_URL:             str  = "http://mlops:8005"
    DASHBOARD_URL:         str  = "http://dashboard:8006"
    AUTH_URL:              str   = "http://auth:8007"

    # ── Feature Engineering ───────────────────────────────────────────────────
    LAG_STEPS:             List[int] = [1, 3, 5]
    SLOPE_WINDOW:          int  = 5
    DEGRADATION_THRESHOLD: float = -3.0
    HO_FLAG_DATARATE_THRESHOLD: float = 5.0

    # ── Monitoring ────────────────────────────────────────────────────────────
    DRIFT_WARNING_THRESHOLD:  float = 0.05
    DRIFT_CRITICAL_THRESHOLD: float = 0.10
    MONITORING_WINDOW_SIZE:   int   = 1000

    # ── CORS ──────────────────────────────────────────────────────────────────
    ALLOWED_ORIGINS: List[str] = ["*"]

    class Config:
        env_file = ".env"


settings = Settings()
