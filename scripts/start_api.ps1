# Inicia la API FastAPI en http://localhost:8080
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$env:PYTHONPATH = $ProjectRoot
$env:MODEL_PATH = "models/resnet18_best.pth"
$env:LABEL_ENCODER_PATH = "models/label_encoder.pkl"
$env:APP_ENV = "development"

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

Write-Host "Iniciando API en http://localhost:8080 ..."
& $Python -m uvicorn src.api.app:app --app-dir $ProjectRoot --host 127.0.0.1 --port 8080 --reload
