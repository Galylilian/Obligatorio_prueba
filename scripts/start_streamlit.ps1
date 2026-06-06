# Inicia Streamlit (requiere API en http://localhost:8080)
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$env:PYTHONPATH = $ProjectRoot
$env:API_URL = "http://localhost:8080"

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

Write-Host "Streamlit en http://localhost:8501 (API: $env:API_URL)"
& $Python -m streamlit run app/streamlit_app.py --server.port 8501
