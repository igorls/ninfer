param([Parameter(Mandatory)][string]$BinDir, [Parameter(Mandatory)][string]$TestDir)
$ErrorActionPreference = 'Stop'
$BinDir = (Resolve-Path -LiteralPath $BinDir).Path
$TestDir = [IO.Path]::GetFullPath($TestDir)
if (Test-Path -LiteralPath $TestDir) { throw 'Use a fresh test directory.' }
New-Item -ItemType Directory -Path $TestDir | Out-Null
$script:checks = 0
function Assert-That([bool]$Condition, [string]$Message) {
    if (!$Condition) { throw $Message }
    $script:checks++
}
function Run-Launcher([string[]]$Arguments, [int]$ExpectedExit) {
    $line = ($Arguments | ForEach-Object { '"' + $_ + '"' }) -join ' '
    $out = Join-Path $TestDir 'output.txt'
    $process = Start-Process -FilePath (Join-Path $BinDir 'ninfer-launcher.exe') -ArgumentList $line `
        -WindowStyle Hidden -PassThru -Wait -RedirectStandardOutput $out
    Assert-That ($process.ExitCode -eq $ExpectedExit) "Unexpected exit $($process.ExitCode): $(Get-Content $out -Raw)"
    return Get-Content -LiteralPath $out -Raw
}
function Fixture([string]$Name, [string]$Model) {
    $path = Join-Path $TestDir $Name
    $directory = [Text.Encoding]::UTF8.GetBytes((@{identity=@{model_id=$Model}} | ConvertTo-Json -Compress))
    $bytes = [byte[]]@(78,73,78,70,69,82,0,2) + [BitConverter]::GetBytes([uint64]$directory.Length) + $directory + [byte[]]::new(64)
    [IO.File]::WriteAllBytes($path, $bytes)
    return $path
}
$probe = Run-Launcher @('--probe') 0 | ConvertFrom-Json
Assert-That $probe.supported 'GPU probe failed.'
$valid = Fixture 'modelo de teste-á.ninfer' 'qwen3.8-27b'
$unsupported = Fixture 'unsupported.ninfer' 'unsupported-model'
$invalid = Join-Path $TestDir 'not-a-model.ninfer'
[IO.File]::WriteAllText($invalid, 'not a model')
foreach ($file in @($unsupported, $invalid)) {
    $data = Join-Path $TestDir ([IO.Path]::GetFileNameWithoutExtension($file))
    $null = Run-Launcher @('--model', $file, '--no-start', '--data-dir', $data) 1
    Assert-That (!(Test-Path -LiteralPath (Join-Path $data 'supervisor.json'))) 'Rejected input created a configuration.'
}
$data = Join-Path $TestDir 'configuração com espaços'
$null = Run-Launcher @('--model',$valid,'--no-start','--data-dir',$data,'--engine-port','18110','--dashboard-port','18099') 0
$configPath = Join-Path $data 'supervisor.json'
$cfg = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
Assert-That ($cfg.engine.args[0] -eq $valid) 'Unicode model path changed.'
Assert-That ($cfg.engine.engine_host -eq '127.0.0.1') 'Engine must default to loopback.'
Assert-That ($cfg.supervisor.host -eq '127.0.0.1') 'Dashboard must default to loopback.'
Assert-That ($cfg.engine.engine_port -eq 18110 -and $cfg.supervisor.port -eq 18099) 'Isolated ports were not preserved.'
Assert-That ($cfg.active_model -eq 'qwen3.8-27b') 'Wrong model identity.'
Assert-That ($cfg.models.Count -eq 1) 'Initial model is missing from the catalog.'
Assert-That ($cfg.engine.executable -eq (Join-Path $BinDir 'ninfer-serve.exe')) 'Wrong installed engine path.'
$before = (Get-FileHash -LiteralPath $configPath).Hash
$null = Run-Launcher @('--model',$valid,'--no-start','--data-dir',$data) 1
Assert-That ((Get-FileHash -LiteralPath $configPath).Hash -eq $before) 'Existing settings were overwritten.'
$null = Run-Launcher @('--no-start','--data-dir',$data) 0
$listener = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, 0)
$listener.Start()
try {
    $port = $listener.LocalEndpoint.Port.ToString()
    $blocked = Join-Path $TestDir 'busy-port'
    $null = Run-Launcher @('--model',$valid,'--no-start','--data-dir',$blocked,'--engine-port',$port,'--dashboard-port','18099') 1
    Assert-That (!(Test-Path -LiteralPath (Join-Path $blocked 'supervisor.json'))) 'Busy port created a configuration.'
} finally { $listener.Stop() }
$null = Run-Launcher @('--no-start','--engine-port','8010','--dashboard-port','8010') 1
Write-Output "Launcher checks passed: $script:checks. No model was loaded."
