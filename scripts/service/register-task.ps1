$ErrorActionPreference = 'Stop'

$taskName = 'NInfer Production Server'
$vbsPath  = 'P:\NInfer\scripts\service\start-supervisor.vbs'
$supExe   = 'P:\NInfer\build-win\apps\ninfer-supervisor\ninfer-supervisor.exe'
$cfgPath  = 'P:\NInfer\supervisor.prod.json'

Write-Host "Registering Scheduled Task '$taskName'..." -ForegroundColor Cyan

if (-not (Test-Path $vbsPath)) {
    throw "Launcher wrapper $vbsPath does not exist."
}

$action = New-ScheduledTaskAction -Execute "C:\WINDOWS\System32\wscript.exe" -Argument "`"$vbsPath`""
$trigger = New-ScheduledTaskTrigger -AtLogOn
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -MultipleInstances IgnoreNew `
    -RestartCount 0 `
    -StartWhenAvailable

# Unregister old task if present
$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Removing existing task registration..."
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
}

Register-ScheduledTask `
    -TaskName $taskName `
    -Action $action `
    -Trigger $trigger `
    -Principal $principal `
    -Settings $settings `
    -Description "Starts and supervises the NInfer production server (:8010) via ninfer-supervisor at user logon." | Out-Null

Write-Host "Scheduled Task '$taskName' registered successfully." -ForegroundColor Green

# Also install HKCU Run key for redundancy
if (Test-Path $supExe) {
    Write-Host "Installing HKCU Run entry for NInferSupervisor..." -ForegroundColor Cyan
    & $supExe --config $cfgPath --install-login
}

Write-Host "`nTask details:" -ForegroundColor Cyan
Get-ScheduledTask -TaskName $taskName | Select-Object TaskName, State, TaskPath | Format-Table -AutoSize
