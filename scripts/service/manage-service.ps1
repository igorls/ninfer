param(
    [ValidateSet('status', 'start', 'stop', 'restart', 'health', 'logs', 'test')]
    [string]$Action = 'status',
    [int]$LogLines = 30
)

$ErrorActionPreference = 'Continue'

$ninferDir   = 'P:\NInfer'
$apiKeyPath  = 'P:\models\ninfer-api-key.txt'
$engineLog   = Join-Path $ninferDir 'supervisor-logs\engine.log'
$autoLog     = Join-Path $ninferDir 'supervisor-logs\autostart.log'
$taskName    = 'NInfer Production Server'
$supUrl      = 'http://127.0.0.1:8099'
$engineUrl   = 'http://127.0.0.1:8010'

$apiKey = ''
if (Test-Path $apiKeyPath) {
    $apiKey = (Get-Content $apiKeyPath -Raw).Trim()
}

function Get-SupervisorStatus {
    try {
        $resp = Invoke-RestMethod -Uri "$supUrl/api/state" -TimeoutSec 3 -ErrorAction Stop
        return $resp
    } catch {
        return $null
    }
}

function Get-EngineHealth {
    try {
        $resp = Invoke-RestMethod -Uri "$engineUrl/health" -TimeoutSec 3 -ErrorAction Stop
        return $resp
    } catch {
        return $null
    }
}

switch ($Action) {
    'status' {
        Write-Host "=== NInfer Service Status ===" -ForegroundColor Cyan
        $supProcs = Get-Process -Name 'ninfer-supervisor' -ErrorAction SilentlyContinue
        $engProcs = Get-Process -Name 'ninfer-serve' -ErrorAction SilentlyContinue

        if ($supProcs) {
            Write-Host "Supervisor Process: RUNNING (PID(s): $(($supProcs | ForEach-Object { $_.Id }) -join ', '))" -ForegroundColor Green
        } else {
            Write-Host "Supervisor Process: STOPPED" -ForegroundColor Yellow
        }

        if ($engProcs) {
            Write-Host "Engine Process:     RUNNING (PID(s): $(($engProcs | ForEach-Object { $_.Id }) -join ', '))" -ForegroundColor Green
        } else {
            Write-Host "Engine Process:     STOPPED" -ForegroundColor Yellow
        }

        $supStat = Get-SupervisorStatus
        if ($supStat) {
            Write-Host "`n--- Supervisor API (:8099) ---" -ForegroundColor Cyan
            Write-Host "  Engine State:   $($supStat.engine.state)"
            Write-Host "  Health:         Status $($supStat.health.status)"
            Write-Host "  Restarts:       $($supStat.engine.restart_count)"
            Write-Host "  Last Event:     $($supStat.engine.last_event)"
            if ($supStat.nvidia_smi -and $supStat.nvidia_smi.total_bytes) {
                $usedGb = [math]::Round($supStat.nvidia_smi.used_bytes / 1GB, 2)
                $totGb  = [math]::Round($supStat.nvidia_smi.total_bytes / 1GB, 2)
                Write-Host "  GPU VRAM:       $usedGb GiB / $totGb GiB"
            }
            if ($supStat.engine_capacity_line) {
                Write-Host "  Capacity:       $($supStat.engine_capacity_line)"
            }
        } else {
            Write-Host "`nSupervisor API (:8099): UNREACHABLE" -ForegroundColor Yellow
        }

        $engHealth = Get-EngineHealth
        if ($engHealth) {
            Write-Host "`n--- Engine Health (:8010) ---" -ForegroundColor Cyan
            Write-Host "  Health:         $($engHealth | ConvertTo-Json -Compress)" -ForegroundColor Green
        } else {
            Write-Host "`nEngine Health (:8010): UNREACHABLE" -ForegroundColor Yellow
        }

        $task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
        if ($task) {
            Write-Host "`n--- Scheduled Task: $taskName ---" -ForegroundColor Cyan
            Write-Host "  State:          $($task.State)"
        }
    }

    'start' {
        Write-Host "Starting NInfer Production Server..." -ForegroundColor Cyan
        $task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
        if ($task) {
            Write-Host "Starting via Scheduled Task '$taskName'..."
            Start-ScheduledTask -TaskName $taskName
        } else {
            Write-Host "Starting via start-supervisor.ps1..."
            & "$ninferDir\scripts\service\start-supervisor.ps1"
        }
        Start-Sleep -Seconds 2
        & $MyInvocation.MyCommand.Path -Action status
    }

    'stop' {
        Write-Host "Stopping NInfer Production Server..." -ForegroundColor Cyan
        $supStat = Get-SupervisorStatus
        if ($supStat) {
            try {
                Write-Host "Sending POST $supUrl/api/stop (header X-NInfer-Supervisor: 1)..."
                Invoke-RestMethod -Uri "$supUrl/api/stop" -Method Post -Headers @{'X-NInfer-Supervisor'='1'} -TimeoutSec 5 | Out-Null
            } catch {
                Write-Host "API stop request note: $_"
            }
        }
        $task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
        if ($task -and $task.State -eq 'Running') {
            Stop-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
        }
        Stop-Process -Name 'ninfer-supervisor', 'ninfer-serve' -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 1
        Write-Host "Stopped." -ForegroundColor Green
    }

    'restart' {
        Write-Host "Restarting NInfer Production Server..." -ForegroundColor Cyan
        $supStat = Get-SupervisorStatus
        if ($supStat) {
            try {
                Write-Host "Sending POST $supUrl/api/restart (header X-NInfer-Supervisor: 1)..."
                Invoke-RestMethod -Uri "$supUrl/api/restart" -Method Post -Headers @{'X-NInfer-Supervisor'='1'} -TimeoutSec 5 | Out-Null
                Start-Sleep -Seconds 3
                & $MyInvocation.MyCommand.Path -Action status
                return
            } catch {
                Write-Host "API restart failed, performing full service restart..."
            }
        }
        & $MyInvocation.MyCommand.Path -Action stop
        Start-Sleep -Seconds 2
        & $MyInvocation.MyCommand.Path -Action start
    }

    'health' {
        Write-Host "Checking Engine Health & Models..." -ForegroundColor Cyan
        $health = Get-EngineHealth
        if ($health) {
            Write-Host "Health: OK -> $($health | ConvertTo-Json -Compress)" -ForegroundColor Green
        } else {
            Write-Host "Health: FAILED / UNREACHABLE" -ForegroundColor Red
        }

        if ($apiKey) {
            try {
                $models = Invoke-RestMethod -Uri "$engineUrl/v1/models" -Headers @{'Authorization'="Bearer $apiKey"} -TimeoutSec 5
                Write-Host "Models: $($models | ConvertTo-Json -Compress)" -ForegroundColor Green
            } catch {
                Write-Host "Models query failed: $_" -ForegroundColor Red
            }
        }
    }

    'test' {
        Write-Host "Running live Chat Completion smoke test..." -ForegroundColor Cyan
        if (-not $apiKey) {
            Write-Host "ERROR: No API key found at $apiKeyPath" -ForegroundColor Red
            exit 1
        }
        $body = @{
            model = "qwen3.8-27b-nvfp4"
            messages = @(
                @{ role = "user"; content = "Hello! State your model name and say ready in 5 words." }
            )
            max_tokens = 64
        } | ConvertTo-Json

        try {
            $resp = Invoke-RestMethod -Uri "$engineUrl/v1/chat/completions" -Method Post `
                -Headers @{'Authorization'="Bearer $apiKey"; 'Content-Type'='application/json'} `
                -Body $body -TimeoutSec 30
            Write-Host "Response received:" -ForegroundColor Green
            Write-Host ($resp | ConvertTo-Json -Depth 5)
        } catch {
            Write-Host "Test request failed: $_" -ForegroundColor Red
        }
    }

    'logs' {
        Write-Host "=== Tail of engine.log ($LogLines lines) ===" -ForegroundColor Cyan
        if (Test-Path $engineLog) {
            Get-Content -Path $engineLog -Tail $LogLines
        } else {
            Write-Host "engine.log does not exist yet." -ForegroundColor Yellow
        }

        Write-Host "`n=== Tail of autostart.log (15 lines) ===" -ForegroundColor Cyan
        if (Test-Path $autoLog) {
            Get-Content -Path $autoLog -Tail 15
        } else {
            Write-Host "autostart.log does not exist yet." -ForegroundColor Yellow
        }
    }
}
