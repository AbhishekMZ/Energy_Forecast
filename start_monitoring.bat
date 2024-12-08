@echo off
echo Starting monitoring infrastructure...

REM Create required directories
mkdir monitoring\prometheus 2>nul
mkdir monitoring\grafana\provisioning\datasources 2>nul
mkdir monitoring\grafana\provisioning\dashboards 2>nul
mkdir monitoring\grafana\dashboards 2>nul

REM Start the services
docker-compose up -d

echo.
echo Monitoring infrastructure is starting up...
echo.
echo Access points:
echo - Grafana: http://localhost:3000 (admin/admin123)
echo - Prometheus: http://localhost:9090
echo - Redis: localhost:6379
echo.
echo Note: It may take a few seconds for all services to be fully operational.
