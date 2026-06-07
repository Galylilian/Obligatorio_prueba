# Levanta API + Streamlit sin Docker (Windows)
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    Write-Error "No existe .venv. Ejecuta: py -3.11 -m venv .venv"
    exit 1
}

$env:PYTHONPATH = $ProjectRoot
$env:API_URL = "http://localhost:8080"
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = "false"
$env:STREAMLIT_SERVER_HEADLESS = "true"

function Test-Port($Port) {
    (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue).Count -gt 0
}

if (-not (Test-Port 8080)) {
    Write-Host "Iniciando API en :8080..."
    Start-Process powershell -WindowStyle Minimized -ArgumentList @(
        "-NoExit", "-Command",
        "cd '$ProjectRoot'; `$env:PYTHONPATH='$ProjectRoot'; & '$Python' -m uvicorn src.api.app:app --app-dir . --host 0.0.0.0 --port 8080"
    )
    Start-Sleep -Seconds 5
} else {
    Write-Host "API ya corre en :8080"
}

if (-not (Test-Port 8501)) {
    Write-Host "Iniciando Streamlit en :8501..."
    Start-Process powershell -WindowStyle Minimized -ArgumentList @(
        "-NoExit", "-Command",
        "cd '$ProjectRoot'; `$env:PYTHONPATH='$ProjectRoot'; `$env:API_URL='http://localhost:8080'; `$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS='false'; & '$Python' -m streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true"
    )
    Start-Sleep -Seconds 4
} else {
    Write-Host "Streamlit ya corre en :8501"
}

Write-Host ""
Write-Host "Streamlit: http://localhost:8501"
Write-Host "API:       http://localhost:8080/docs"
Write-Host "KPIs JSON: http://localhost:8080/dashboard/stats"
Write-Host ""
Write-Host "Grafana (:3000) requiere Docker. Si falla, ejecuta como Admin:"
Write-Host "  powershell -ExecutionPolicy Bypass -File .\scripts\fix_wsl_docker.ps1"
