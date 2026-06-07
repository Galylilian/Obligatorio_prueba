# Sincroniza metricas del modelo (accuracy, F1, etc.) a JSON + PostgreSQL/SQLite
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$env:PYTHONPATH = $ProjectRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

Write-Host "Evaluando modelo y guardando metricas..."
& $Python -m src.core.evaluate
Write-Host ""
Write-Host "Listo. Refresca Grafana: http://localhost:3000"
