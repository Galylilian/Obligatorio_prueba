# Inicia stack analitico: PostgreSQL + API + Grafana
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Creado .env desde .env.example"
}

Write-Host "Levantando PostgreSQL + API + Grafana..."
$env:DATABASE_URL = "postgresql+psycopg2://fallapp:fallapp@db:5432/fall_analytics"
docker compose --profile analytics up --build -d db api grafana

Write-Host ""
Write-Host "API:     http://localhost:8080/docs"
Write-Host "Stats:   http://localhost:8080/dashboard/stats"
Write-Host "Grafana: http://localhost:3000  (admin / admin por defecto)"
Write-Host ""
Write-Host "Dashboard: Deteccion de caidas - KPIs"
