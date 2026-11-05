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
    Assert-That ($process.ExitCode -eq $ExpectedExit) "Unexpected exit $($process.ExitCode): $(Get-Content $out -Raw -Encoding UTF8)"
    return Get-Content -LiteralPath $out -Raw -Encoding UTF8
}
function Assert-ServeArguments([string[]]$Arguments, [string]$Name) {
    # Exercise the real product parser without loading the metadata-only model fixture.
    # A nonexistent authentication file is resolved only after all arguments are parsed.
    $missingKey = Join-Path $TestDir 'missing-admission-key.txt'
    $line = (@($Arguments) + @('--api-key-file', $missingKey) | ForEach-Object { '"' + $_ + '"' }) -join ' '
    $out = Join-Path $TestDir "$Name-serve.stdout"
    $err = Join-Path $TestDir "$Name-serve.stderr"
    $process = Start-Process -FilePath (Join-Path $BinDir 'ninfer-serve.exe') -ArgumentList $line `
        -WindowStyle Hidden -PassThru -Wait -RedirectStandardOutput $out -RedirectStandardError $err
    $errorText = Get-Content -LiteralPath $err -Raw -Encoding UTF8
    Assert-That ($process.ExitCode -eq 1 -and $errorText.Contains('server.api_key_file: cannot read the API key file')) `
        "Generated $Name arguments failed before expected authentication admission: $errorText"
}
function Fixture([string]$Name, [string]$Model, [bool]$Mtp = $true, [string]$MtpArchitecture = 'Qwen3_5MTP') {
    $path = Join-Path $TestDir $Name
    $components = @{text=@{config=@{architectures=@('Qwen3_5ForCausalLM');model_type='qwen3_5_text'}}}
    if ($Mtp) { $components.mtp=@{config=@{architectures=@($MtpArchitecture)}} }
    $root=@{components=$components}
    if ($Model) { $root.metadata=@{name=$Model} }
    $directory = [Text.Encoding]::UTF8.GetBytes(($root | ConvertTo-Json -Depth 8 -Compress))
    $bytes = [byte[]]@(78,73,78,70,69,82,0,3) + [BitConverter]::GetBytes([uint64]$directory.Length) + [byte[]]::new(16) + $directory
    $payloadStart = [int]([Math]::Ceiling($bytes.Length / 4096.0) * 4096)
    $bytes += [byte[]]::new($payloadStart - $bytes.Length + 64)
    [IO.File]::WriteAllBytes($path, $bytes)
    return $path
}
$probe = Run-Launcher @('--probe') 0 | ConvertFrom-Json
Assert-That $probe.supported 'GPU probe failed.'
$valid = Fixture ('modelo de teste-' + [char]0xE1 + '.ninfer') 'qwen3.8-27b'
$legacy = Fixture 'legacy.ninfer' 'legacy-model'
$legacyBytes = [IO.File]::ReadAllBytes($legacy)
$legacyBytes[7]=2
[IO.File]::WriteAllBytes($legacy,$legacyBytes)
$truncated = Join-Path $TestDir 'truncated.ninfer'
[IO.File]::WriteAllBytes($truncated,([byte[]]@(78,73,78,70,69,82,0,3) + [BitConverter]::GetBytes([uint64]4096) + [byte[]]::new(16)))
$invalid = Join-Path $TestDir 'not-a-model.ninfer'
[IO.File]::WriteAllText($invalid, 'not a model')
foreach ($file in @($legacy, $truncated, $invalid)) {
    $data = Join-Path $TestDir ([IO.Path]::GetFileNameWithoutExtension($file))
    $null = Run-Launcher @('--model', $file, '--no-start', '--data-dir', $data) 1
    Assert-That (!(Test-Path -LiteralPath (Join-Path $data 'supervisor.json'))) 'Rejected input created a configuration.'
}
$data = Join-Path $TestDir ('config with spaces-' + [char]0xE7)
$null = Run-Launcher @('--model',$valid,'--no-start','--data-dir',$data,'--engine-port','18110','--dashboard-port','18099') 0
$configPath = Join-Path $data 'supervisor.json'
$cfg = Get-Content -LiteralPath $configPath -Raw -Encoding UTF8 | ConvertFrom-Json
Assert-That ($cfg.engine.args[0] -eq $valid) 'Unicode model path changed.'
Assert-That ($cfg.engine.engine_host -eq '127.0.0.1') 'Engine must default to loopback.'
Assert-That ($cfg.supervisor.host -eq '127.0.0.1') 'Dashboard must default to loopback.'
Assert-That ($cfg.engine.engine_port -eq 18110 -and $cfg.supervisor.port -eq 18099) 'Isolated ports were not preserved.'
Assert-That ($cfg.active_model -eq 'qwen3.8-27b') 'Wrong model identity.'
Assert-That ($cfg.models.Count -eq 1) 'Initial model is missing from the catalog.'
$headIndex=[Array]::IndexOf($cfg.engine.args,'--proposal-head')
Assert-That ($headIndex -ge 0 -and $cfg.engine.args[$headIndex+1] -eq 'optimized') 'MTP profile omitted its optimized proposal head.'
Assert-That ($cfg.engine.args -notcontains '--gdn-state-dtype') 'Setup emitted a removed engine flag.'
$reserveIndex=[Array]::IndexOf($cfg.engine.args,'--desktop-reserve')
Assert-That ($reserveIndex -ge 0 -and $cfg.engine.args[$reserveIndex+1] -eq '8GiB') 'Setup must preserve the default desktop reserve.'
Assert-ServeArguments $cfg.engine.args 'dense-mtp'
Assert-That ($cfg.engine.executable -eq (Join-Path $BinDir 'ninfer-serve.exe')) 'Wrong installed engine path.'
$before = (Get-FileHash -LiteralPath $configPath).Hash
$null = Run-Launcher @('--model',$valid,'--no-start','--data-dir',$data) 1
Assert-That ((Get-FileHash -LiteralPath $configPath).Hash -eq $before) 'Existing settings were overwritten.'
$null = Run-Launcher @('--no-start','--data-dir',$data) 0
$recipe = Fixture 'custom recipe.ninfer' 'custom-checkpoint-recipe' $false
$recipeData = Join-Path $TestDir 'recipe-config'
$null = Run-Launcher @('--model',$recipe,'--no-start','--data-dir',$recipeData,'--engine-port','18110','--dashboard-port','18099') 0
$recipeConfig=Get-Content -LiteralPath (Join-Path $recipeData 'supervisor.json') -Raw -Encoding UTF8 | ConvertFrom-Json
Assert-That ($recipeConfig.active_model -eq 'custom-checkpoint-recipe') 'Recipe metadata was not preserved.'
Assert-That ($recipeConfig.engine.args -contains 'none' -and $recipeConfig.engine.args -notcontains '--draft-tokens') 'Non-MTP artifact requested speculative execution.'
Assert-ServeArguments $recipeConfig.engine.args 'recipe-none'
$unnamed = Fixture 'unnamed.ninfer' '' $false
$null = Run-Launcher @('--model',$unnamed,'--no-start','--data-dir',(Join-Path $TestDir 'unnamed-config'),'--engine-port','18110','--dashboard-port','18099') 0
$unnamedConfig=Get-Content -LiteralPath (Join-Path $TestDir 'unnamed-config/supervisor.json') -Raw -Encoding UTF8 | ConvertFrom-Json
Assert-That ($unnamedConfig.active_model -eq 'unnamed') 'Optional name did not fall back to the filename.'
$flash = Fixture 'flash recipe.ninfer' 'custom-flash-recipe' $true 'Qwen4ExpMTP'
$null = Run-Launcher @('--model',$flash,'--no-start','--data-dir',(Join-Path $TestDir 'flash-config'),'--engine-port','18110','--dashboard-port','18099') 0
$flashConfig=Get-Content -LiteralPath (Join-Path $TestDir 'flash-config/supervisor.json') -Raw -Encoding UTF8 | ConvertFrom-Json
Assert-That ($flashConfig.engine.args -contains 'mtp' -and $flashConfig.engine.args -contains '--draft-tokens') 'Flash profile omitted MTP.'
$flashHeadIndex=[Array]::IndexOf($flashConfig.engine.args,'--proposal-head')
Assert-That ($flashHeadIndex -lt 0 -or $flashConfig.engine.args[$flashHeadIndex+1] -eq 'full') 'Flash profile requested its unsupported optimized head.'
Assert-ServeArguments $flashConfig.engine.args 'flash-mtp'
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
