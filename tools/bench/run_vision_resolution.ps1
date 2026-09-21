[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$SourceImage,
    [string]$Url = 'http://127.0.0.1:8010/v1/chat/completions',
    [string]$RequestLog = "$env:LOCALAPPDATA/NInfer/logs/requests.jsonl",
    [string]$OutputDir = 'profiles/bench/vision-resolution',
    [int[]]$Widths = @(320, 640, 960, 1280, 1600, 1972, 2240),
    [int]$Repetitions = 3,
    [int]$MaxTokens = 1
)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
$sourcePath = (Resolve-Path -LiteralPath $SourceImage).Path
$outputPath = [IO.Path]::GetFullPath($OutputDir)
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
$source = [Drawing.Image]::FromFile($sourcePath)
$cases = @()
try {
    foreach ($width in $Widths) {
        $height = [Math]::Max(1, [int][Math]::Round($source.Height * $width / $source.Width))
        $path = Join-Path $outputPath ("fixture-{0}x{1}.png" -f $width,$height)
        $bitmap = [Drawing.Bitmap]::new($width, $height, [Drawing.Imaging.PixelFormat]::Format24bppRgb)
        $graphics = [Drawing.Graphics]::FromImage($bitmap)
        try {
            $graphics.InterpolationMode = [Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
            $graphics.PixelOffsetMode = [Drawing.Drawing2D.PixelOffsetMode]::HighQuality
            $graphics.DrawImage($source, [Drawing.Rectangle]::new(0, 0, $width, $height))
            $bitmap.Save($path, [Drawing.Imaging.ImageFormat]::Png)
        } finally {
            $graphics.Dispose()
            $bitmap.Dispose()
        }
        $cases += [pscustomobject]@{Width=$width;Height=$height;Path=$path}
    }
} finally { $source.Dispose() }

$clientTag = 'NInfer-Vision-Resolution-Bench/1'
$started = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
$runNonce = [Guid]::NewGuid().ToString('N')
$order = for ($rep = 0; $rep -lt $Repetitions; $rep++) {
    if ($rep % 2 -eq 0) { $cases } else { @($cases | Sort-Object Width -Descending) }
}
$wall = @()
$index = 0
foreach ($case in $order) {
    $index++
    $bytes = [IO.File]::ReadAllBytes($case.Path)
    $uri = 'data:image/png;base64,' + [Convert]::ToBase64String($bytes)
    $body = @{
        model='qwen3.8-flash-next'
        messages=@(@{role='user';content=@(
            @{type='image_url';image_url=@{url=$uri;detail='auto'}},
            @{type='text';text=("Resolution benchmark {0}-{1}. Describe the dominant layout in one token." -f $runNonce,$index)}
        )})
        max_tokens=$MaxTokens
        temperature=0
        chat_template_kwargs=@{enable_thinking=$false}
    } | ConvertTo-Json -Depth 8 -Compress
    $watch = [Diagnostics.Stopwatch]::StartNew()
    $response = Invoke-RestMethod -Uri $Url -Method Post -ContentType 'application/json' `
        -UserAgent $clientTag -Body $body -TimeoutSec 120
    $watch.Stop()
    $wall += [pscustomobject]@{width=$case.Width;height=$case.Height;wall_seconds=$watch.Elapsed.TotalSeconds}
}

$events = Get-Content -LiteralPath $RequestLog -Tail ([Math]::Max(500, $order.Count * 8)) |
    ForEach-Object { try { $_ | ConvertFrom-Json } catch {} } |
    Where-Object { $_.timestamp_unix_ms -ge $started -and $_.request.client -eq $clientTag -and
                   $_.event -in @('request_start','request_done') }
$starts = @{}; $done = @{}
foreach ($event in $events) {
    if ($event.event -eq 'request_start') { $starts[[int]$event.request.request_id] = $event }
    else { $done[[int]$event.request.request_id] = $event }
}
if ($starts.Count -ne $order.Count -or $done.Count -ne $order.Count) {
    throw "Expected $($order.Count) start/done pairs, found $($starts.Count)/$($done.Count)"
}
$rows = @(); $sequence = 0
foreach ($id in @($starts.Keys | Sort-Object)) {
    $case = $order[$sequence]; $start = $starts[$id]; $finish = $done[$id]
    $prep = $start.preparation_seconds; $timing = $finish.timings_seconds
    $rows += [pscustomobject]@{
        request_id=$id; width=$case.Width; height=$case.Height
        source_pixels=$case.Width*$case.Height; media_bytes=$prep.media_bytes
        raw_patches=$prep.raw_patches; vision_tokens=$prep.vision_tokens
        cache_hit=$prep.cache_hits -gt 0; prepare_seconds=$timing.prepare
        preprocess_work_seconds=$prep.media_preprocess_work
        vision_seconds=$timing.vision; prefill_seconds=$timing.prefill
        ttft_seconds=$timing.ttft; queue_wait_seconds=$finish.engine_timing.queue_wait_seconds
        completion_tokens=$finish.result.completion_tokens; decode_seconds=$timing.decode
        decode_tokens_per_second=if ($timing.decode -gt 0) {$finish.result.completion_tokens/$timing.decode} else {0}
        wall_seconds=$wall[$sequence].wall_seconds
    }
    $sequence++
}
$csv = Join-Path $outputPath 'samples.csv'
$rows | Export-Csv -LiteralPath $csv -NoTypeInformation
$summary = $rows | Group-Object width,height,source_pixels,raw_patches,vision_tokens | ForEach-Object {
    $g=$_.Group
    [pscustomobject]@{
        width=$g[0].width; height=$g[0].height; source_pixels=$g[0].source_pixels
        raw_patches=$g[0].raw_patches; vision_tokens=$g[0].vision_tokens
        cold_prepare_ms=[Math]::Round(($g|Where-Object {!$_.cache_hit}|Measure-Object prepare_seconds -Average).Average*1000,2)
        median_vision_ms=[Math]::Round(($g.vision_seconds|Sort-Object)[[int][Math]::Floor($g.Count/2)]*1000,2)
        median_prefill_ms=[Math]::Round(($g.prefill_seconds|Sort-Object)[[int][Math]::Floor($g.Count/2)]*1000,2)
        median_ttft_ms=[Math]::Round(($g.ttft_seconds|Sort-Object)[[int][Math]::Floor($g.Count/2)]*1000,2)
        median_queue_ms=[Math]::Round(($g.queue_wait_seconds|Sort-Object)[[int][Math]::Floor($g.Count/2)]*1000,2)
        median_decode_tokens_per_second=[Math]::Round(($g.decode_tokens_per_second|Sort-Object)[[int][Math]::Floor($g.Count/2)],2)
    }
} | Sort-Object width
$summary | Export-Csv -LiteralPath (Join-Path $outputPath 'summary.csv') -NoTypeInformation
$summary | Format-Table -AutoSize
