[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$BuildDir,
    [Parameter(Mandatory)][string]$LauncherBuildDir,
    [Parameter(Mandatory)][string]$Iscc,
    [Parameter(Mandatory)][string]$FFmpegLicense,
    [Parameter(Mandatory)][string]$FFmpegSourceNote,
    [Parameter(Mandatory)][string]$CurlLicense,
    [string]$Version = '2026.09.11-preview.1',
    [string]$OutputDir = (Join-Path $PSScriptRoot '../../dist')
)
$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$BuildDir = (Resolve-Path -LiteralPath $BuildDir).Path
$LauncherBuildDir = (Resolve-Path -LiteralPath $LauncherBuildDir).Path
$OutputDir = [IO.Path]::GetFullPath($OutputDir)
$payload = Join-Path $OutputDir ('payload-' + $Version)
if (Test-Path -LiteralPath $payload) { throw "Use a new version or remove the old staging directory first: $payload" }
New-Item -ItemType Directory -Path (Join-Path $payload 'bin'),(Join-Path $payload 'licenses') -Force | Out-Null
& cmake "-DDESTINATION=$payload/bin" -P (Join-Path $BuildDir 'windows-package-Release.cmake')
if ($LASTEXITCODE -ne 0) { throw 'Runtime dependency packaging failed.' }
Copy-Item -LiteralPath (Join-Path $LauncherBuildDir 'Release/ninfer-launcher.exe') -Destination (Join-Path $payload 'bin')
Copy-Item -LiteralPath (Join-Path $repo 'LICENSE') -Destination $payload
$licenses = @{
    'FFmpeg.txt' = $FFmpegLicense
    'FFmpeg-SOURCE.txt' = $FFmpegSourceNote
    'curl.txt' = $CurlLicense
    'nlohmann-json.txt' = (Join-Path $repo 'third_party/nlohmann/LICENSE.MIT')
    'cpp-httplib.txt' = (Join-Path $repo 'third_party/cpp-httplib/LICENSE')
    'spdlog.txt' = (Join-Path $repo 'third_party/spdlog/LICENSE')
    'utf8proc.txt' = (Join-Path $repo 'third_party/utf8proc/LICENSE.md')
    'xgrammar.txt' = (Join-Path $repo 'third_party/xgrammar/LICENSE')
    'dlpack.txt' = (Join-Path $repo 'third_party/xgrammar/3rdparty/dlpack/LICENSE')
}
foreach ($entry in $licenses.GetEnumerator()) {
    Copy-Item -LiteralPath $entry.Value -Destination (Join-Path $payload ('licenses/' + $entry.Key))
}
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'installer-notes.txt') -Destination (Join-Path $payload 'INSTALL-NOTES.txt')
$files = @(Get-ChildItem -LiteralPath (Join-Path $payload 'bin') -File | ForEach-Object {
    @{name=$_.Name; bytes=$_.Length; sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()}
})
@{version=$Version; source_commit=(& git -C $repo rev-parse HEAD); architecture='sm_120a'; signed=$false; files=$files} |
    ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $payload 'release-manifest.json') -Encoding utf8

# Check every application without relying on a developer PATH. No model is loaded.
$originalPath = $env:PATH
try {
    $env:PATH = "$env:SystemRoot/System32;$env:SystemRoot"
    foreach ($app in @('ninfer.exe','ninfer-serve.exe','ninfer-supervisor.exe','ninfer-launcher.exe')) {
        $process = Start-Process -FilePath (Join-Path $payload "bin/$app") -ArgumentList '--help' -WindowStyle Hidden -PassThru -Wait
        if ($process.ExitCode -ne 0) { throw "$app cannot start with the bundled runtime: $($process.ExitCode)" }
    }
} finally { $env:PATH = $originalPath }
& $Iscc "/DPayloadDir=$payload" "/DReleaseVersion=$Version" "/DOutputDir=$OutputDir" (Join-Path $PSScriptRoot 'installer.iss')
if ($LASTEXITCODE -ne 0) { throw 'Installer compilation failed.' }
$installer = Join-Path $OutputDir "NInfer-$Version-windows-x64-unsigned.exe"
$digest = (Get-FileHash -LiteralPath $installer -Algorithm SHA256).Hash.ToLowerInvariant()
"$digest  $([IO.Path]::GetFileName($installer))" | Set-Content -LiteralPath (Join-Path $OutputDir "NInfer-$Version-SHA256SUMS.txt") -Encoding ascii
Write-Output $installer
