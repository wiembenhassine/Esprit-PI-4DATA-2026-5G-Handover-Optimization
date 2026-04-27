#!/bin/bash
# ── AI Handover Optimization Platform — Launch All Services ──
set -e

PROJ=$(cd "$(dirname "$0")" && pwd)
cd "$PROJ"

echo "🚀 Starting AI Handover Optimization Platform..."
echo "   Models: model_registry/ (9 artefacts)"
echo ""

# Auth must start first (all other services depend on JWT verification)
uvicorn services.auth.main:app           --port 8007 --log-level warning &
sleep 1   # give auth time to bind

uvicorn services.data_ingestion.main:app  --port 8001 --log-level warning &
uvicorn services.data_processing.main:app --port 8002 --log-level warning &
uvicorn services.prediction.main:app      --port 8003 --log-level warning &
uvicorn services.monitoring.main:app      --port 8004 --log-level warning &
uvicorn services.mlops.main:app           --port 8005 --log-level warning &
uvicorn services.dashboard.main:app       --port 8006 --log-level warning &
sleep 2   # wait for upstream services before starting gateway

uvicorn services.api_gateway.main:app     --port 8000 --log-level warning &

echo ""
echo "✅ All 8 services started."
echo ""
echo "   API Gateway (main entry):  http://localhost:8000/docs"
echo "   Auth Service:              http://localhost:8007/docs"
echo "   Prediction (real models):  http://localhost:8003/docs"
echo ""
echo "   Credentials:"
echo "     operator1  / operator_pass"
echo "     scientist1 / scientist_pass"
echo "     admin      / admin_pass"
echo ""
echo "   Set KERAS_ENSEMBLE=true to enable XGB+NN averaging."
echo ""
echo "   Press Ctrl+C to stop all services."
wait
