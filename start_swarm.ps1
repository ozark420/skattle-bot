# TradingSwarmCore Startup Script
# Starts both the trading agent (via watchdog) and the dashboard

Write-Host "[SWARM] Starting Trading Swarm Core..." -ForegroundColor Cyan
Write-Host ""

$SwarmDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Load environment variables from .env file
$envFile = Join-Path $SwarmDir ".env"
if (Test-Path $envFile) {
    Write-Host "[OK] Loading API keys from .env..." -ForegroundColor Yellow
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
            $name = $matches[1].Trim()
            $value = $matches[2].Trim()
            [Environment]::SetEnvironmentVariable($name, $value, "Process")
        }
    }
} else {
    Write-Host "[ERROR] No .env file found! Create one with your API keys." -ForegroundColor Red
    exit 1
}

# Check Python
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    Write-Host "[ERROR] Python not found. Please install Python 3.10+" -ForegroundColor Red
    exit 1
}

# Install dependencies if needed
Write-Host "[OK] Checking dependencies..." -ForegroundColor Yellow
Push-Location $SwarmDir
pip install -q -r requirements.txt 2>$null
Pop-Location

# Start Dashboard in background
Write-Host "[OK] Starting Dashboard Server (port 8420)..." -ForegroundColor Yellow
$dashboardJob = Start-Job -ScriptBlock {
    param($dir, $bankrKey, $polygonKey, $newsKey)
    [Environment]::SetEnvironmentVariable("BANKR_API_KEY", $bankrKey, "Process")
    [Environment]::SetEnvironmentVariable("POLYGON_API_KEY", $polygonKey, "Process")
    [Environment]::SetEnvironmentVariable("NEWS_API_KEY", $newsKey, "Process")
    Set-Location $dir
    python dashboard/server.py
} -ArgumentList $SwarmDir, $env:BANKR_API_KEY, $env:POLYGON_API_KEY, $env:NEWS_API_KEY

Start-Sleep -Seconds 2

# Start Watchdog (foreground)
Write-Host "[OK] Starting Watchdog + Trading Agent..." -ForegroundColor Yellow
Write-Host ""
Write-Host "===========================================================" -ForegroundColor DarkGray
Write-Host "  Dashboard: http://localhost:8420" -ForegroundColor Green
Write-Host "  Press Ctrl+C to stop everything" -ForegroundColor DarkGray
Write-Host "===========================================================" -ForegroundColor DarkGray
Write-Host ""

try {
    Push-Location $SwarmDir
    python watchdog.py
}
finally {
    Write-Host ""
    Write-Host "Stopping dashboard..." -ForegroundColor Yellow
    Stop-Job $dashboardJob -ErrorAction SilentlyContinue
    Remove-Job $dashboardJob -ErrorAction SilentlyContinue
    Pop-Location
    Write-Host "[STOPPED] Swarm stopped." -ForegroundColor Red
}
