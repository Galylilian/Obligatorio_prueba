# Repara WSL + Docker Desktop (requiere ejecutar como Administrador)
# Uso: clic derecho -> "Ejecutar con PowerShell" o:
#   powershell -ExecutionPolicy Bypass -File .\scripts\fix_wsl_docker.ps1

$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator
)

if (-not $isAdmin) {
    Write-Host "Solicitando permisos de administrador..."
    Start-Process powershell.exe -Verb RunAs -ArgumentList @(
        "-ExecutionPolicy", "Bypass",
        "-File", $MyInvocation.MyCommand.Path
    )
    exit
}

Write-Host "=== Reparacion WSL + Docker ===" -ForegroundColor Cyan

Write-Host "`n[1/5] Habilitando componentes de Windows..."
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart | Out-Null
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart | Out-Null

Write-Host "[2/5] Actualizando WSL..."
wsl --update --web-download 2>&1

Write-Host "[3/5] Instalando Ubuntu (si no existe)..."
$distros = wsl -l -q 2>$null
if ($distros -notmatch "Ubuntu") {
    wsl --install -d Ubuntu 2>&1
} else {
    Write-Host "Ubuntu ya instalado."
}

Write-Host "[4/5] WSL version por defecto = 2..."
wsl --set-default-version 2 2>&1

Write-Host "[5/5] Iniciando Docker Desktop..."
$dockerExe = "${env:ProgramFiles}\Docker\Docker\Docker Desktop.exe"
if (Test-Path $dockerExe) {
    Start-Process $dockerExe
    Write-Host "Esperando 60 segundos a que Docker arranque..."
    Start-Sleep -Seconds 60
    docker info 2>&1 | Select-Object -First 8
} else {
    Write-Host "Docker Desktop no encontrado. Instalalo desde https://www.docker.com/products/docker-desktop/" -ForegroundColor Yellow
}

Write-Host "`n=== IMPORTANTE ===" -ForegroundColor Yellow
Write-Host "Si es la primera vez, REINICIA la PC y abre Docker Desktop manualmente."
Write-Host "Luego ejecuta: .\scripts\start_analytics.ps1"
Write-Host "Grafana: http://localhost:3000 (admin / admin)"
Read-Host "Presiona Enter para cerrar"
