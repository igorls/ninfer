param(
    [switch]$Foreground
)

$ErrorActionPreference = 'Continue'

$ninferDir     = 'P:\NInfer'
$configPath    = Join-Path $ninferDir 'supervisor.prod.json'
$supervisorExe = Join-Path $ninferDir 'build-win\apps\ninfer-supervisor\ninfer-supervisor.exe'
$engineExe     = Join-Path $ninferDir 'build-win\apps\ninfer-serve.exe'
$modelPath     = 'P:\models\qwen3_8_27b_nvfp4.ninfer'
$apiKeyPath    = 'P:\models\ninfer-api-key.txt'
$logsDir       = Join-Path $ninferDir 'supervisor-logs'
$logFile       = Join-Path $logsDir 'autostart.log'
$portEngine    = 8010
$portSup       = 8099

if (-not (Test-Path $logsDir)) {
    New-Item -ItemType Directory -Path $logsDir -Force | Out-Null
}

function Log-Msg($msg) {
    $ts = (Get-Date -Format o)
    Add-Content -Path $logFile -Value "$ts [launcher] $msg"
    if ($Foreground) {
        Write-Host "$ts [launcher] $msg"
    }
}

function Port-Busy($p) {
    try {
        $conn = Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue
        return [bool]$conn
    } catch {
        return $false
    }
}

Log-Msg "NInfer production service launcher starting (PID $PID, Foreground=$Foreground)"

if (-not (Test-Path $supervisorExe)) {
    Log-Msg "ERROR: Supervisor executable not found: $supervisorExe"
    exit 1
}
if (-not (Test-Path $engineExe)) {
    Log-Msg "ERROR: Engine executable not found: $engineExe"
    exit 1
}
if (-not (Test-Path $modelPath)) {
    Log-Msg "ERROR: Model artifact not found: $modelPath"
    exit 1
}
if (-not (Test-Path $apiKeyPath)) {
    Log-Msg "ERROR: API key file not found: $apiKeyPath"
    exit 1
}
if (-not (Test-Path $configPath)) {
    Log-Msg "ERROR: Supervisor config not found: $configPath"
    exit 1
}

if (Port-Busy $portSup) {
    Log-Msg "Port $portSup is already active (supervisor already running). Exiting."
    exit 0
}

# Rotate autostart log if it exceeds 4MB
if ((Test-Path $logFile) -and ((Get-Item $logFile).Length -gt 4MB)) {
    $prev = "$logFile.prev"
    if (Test-Path $prev) { Remove-Item $prev -Force -ErrorAction SilentlyContinue }
    Rename-Item $logFile $prev -ErrorAction SilentlyContinue
}

if ($Foreground) {
    Log-Msg "Starting supervisor in foreground: $supervisorExe --config $configPath"
    & $supervisorExe --config $configPath
    $code = $LASTEXITCODE
    Log-Msg "Supervisor exited with code $code"
    exit $code
} else {
    Log-Msg "Launching supervisor detached: $supervisorExe --config $configPath"
    $proc = Start-Process -FilePath $supervisorExe -ArgumentList "--config `"$configPath`"" -WorkingDirectory $ninferDir -WindowStyle Hidden -PassThru
    Log-Msg "Supervisor launched with PID $($proc.Id)"
    exit 0
}
