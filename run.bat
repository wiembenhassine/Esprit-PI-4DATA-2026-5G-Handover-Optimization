@echo off
echo ============================================================
echo  5G Handover Optimization Platform
echo ============================================================
echo.

REM Check Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker is not running.
    echo Please start Docker Desktop and wait until it says "Engine running".
    pause
    exit /b 1
)

echo [1/3] Cleaning up old Docker data to free space...
docker compose down --remove-orphans 2>nul
docker builder prune -f >nul 2>&1
echo       Done.
echo.

echo [2/3] Building base image (installs Python packages - first run takes 5-10 min)...
docker compose build base
if errorlevel 1 (
    echo ERROR: Base image build failed. Check your internet connection.
    pause
    exit /b 1
)
echo       Done.
echo.

echo [3/3] Building and starting all services...
docker compose up --build -d
if errorlevel 1 (
    echo ERROR: Failed to start services.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  All services starting up...
echo  (Prediction service takes ~90s to load the models)
echo.
echo  Frontend:   http://localhost
echo  API docs:   http://localhost:8000/docs
echo.
echo  Login: operator1 / operator_pass
echo         scientist1 / scientist_pass
echo         admin / admin_pass
echo.
echo  Run "docker compose logs -f" to watch logs
echo  Run "docker compose down" to stop
echo ============================================================
pause
