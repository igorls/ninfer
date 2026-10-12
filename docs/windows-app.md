# Windows app

For image/video input, use a media-enabled build. FFmpeg must include its C/C++ headers and
MSVC import libraries, as well as the matching runtime DLLs; an `ffmpeg.exe` command-line
installation alone is insufficient. A shared Windows build linked from
[FFmpeg's download page](https://ffmpeg.org/download.html) supplies those files. Build or install
a shared libcurl with headers and an MSVC import library; Schannel uses Windows' native TLS and
certificate store. Set the dependency locations for your machine:

```powershell
$env:PATH = "C:/deps/ffmpeg/bin;C:/deps/curl/bin;$env:PATH"
cmake -S . -B build-win -DNINFER_BUILD_MEDIA=ON `
  -DFFMPEG_ROOT=C:/deps/ffmpeg -DCURL_ROOT=C:/deps/curl
```

Add `--vision` to the configured model's engine arguments to load its Vision allocations.
The media-enabled binaries still accept text-only profiles without that flag. The installer
packages the discovered media runtime DLLs alongside the applications, so the installed app does
not depend on a developer shell's `PATH`. After updating an existing installation, change the
installed configuration through the dashboard or its JSON file and restart the engine.

Build and install for the current Windows user, without administrator access:

```powershell
cmake --build build-win --config Release --target ninfer ninfer-serve ninfer-supervisor -j
.\scripts\windows\install.ps1 -ConfigPath .\supervisor.local.json
```

The configuration must describe the model artifacts already present on this machine. Relative
artifact and API-key paths are resolved against its original working directory. Models are not
copied or downloaded. The installed engine works from the app's data directory; use absolute
paths for any other auxiliary resources in custom arguments.

| Item | Default location |
|---|---|
| Executables and runtime DLLs | `%LOCALAPPDATA%\Programs\NInfer\bin` |
| Maintained chat templates for `--chat-template` | `%LOCALAPPDATA%\Programs\NInfer\chat-templates` |
| Active configuration | `%LOCALAPPDATA%\NInfer\supervisor.json` |
| Tray preferences | `%LOCALAPPDATA%\NInfer\supervisor.tray.json` |
| Engine and request logs | `%LOCALAPPDATA%\NInfer\logs` |
| Application shortcuts | Start menu → NInfer |

The dashboard's request figures and its Diagnose findings come from the request log. The
log is never trimmed, so the Supervisor reads an existing one once, in the background, when it
starts (about 8 s for 750 MiB) and the findings show that progress until it finishes; after that
it reads only the records the engine appends.

**Calling apps** on Monitor reports observed requests over a rolling 15-minute window, grouped by
User-Agent. Completed, cancelled, failed, unfinished-at-restart and awaiting-outcome requests are separate; admission
timeouts are counted explicitly. First-token, queue and prefill timings include means,
p95 values; an app whose queue wait averages a second or more is shown in amber, and requests
that expired before admission add a caution line. Queue timing is labelled as first-output or
whole-request according to the recorded boundary. Missing measurements remain unavailable.

Reuse is the recorded cache-hit token count divided by prompt tokens; computed tokens
include new input and recovery work. Full prefills do not diagnose changed conversations
or tool definitions. The evidence disclosure reports explicit admission fallback reasons,
disabled reuse, sample counts, unrecorded causes and engine-instance boundaries. Requests are joined by
engine instance and request ID, so restarting the engine cannot mix client attribution.
Diagnose also uses recorded queue timings; request total already includes queue
wait and must not be subtracted from start/done wall time to infer it.

The installed configuration is the active authority. Changes to the original source JSON do not
alter the running installation. The dashboard saves model switches and settings to the installed
copy. `-InstallDir` and `-DataDir` select custom app/data directories; keep them separate.

The setup and management scripts detect packaged callers such as Codex and relaunch through the
Windows desktop. This prevents Windows from redirecting app data and registration into the
caller's private MSIX storage. A configuration from a previous redirected installation is used
only when the desktop installation has no configuration yet.

NInfer starts at sign-in and displays a tray icon. Closing a browser, terminal, editor, or coding
agent does not stop it. **Exit** in the tray closes Supervisor and its managed engine. Signing out
also closes the app; running before sign-in would require a separate Windows service design.
Toggle **Start at login** in the tray to change the sign-in preference.

Launch from the Start menu, or use the installed binary from automation:

```powershell
& "$env:LOCALAPPDATA\Programs\NInfer\bin\ninfer-supervisor.exe" `
  --config "$env:LOCALAPPDATA\NInfer\supervisor.json" --background
```

`--background` asks the running Explorer desktop to create the app. Directly spawning a GUI
executable from an agent terminal can inherit that terminal's process job, even with a hidden
window. The desktop launch requires an interactive Windows desktop; it fails explicitly if the
desktop broker is unavailable. See Microsoft's [desktop launch pattern](https://devblogs.microsoft.com/oldnewthing/20131118-00/?p=2643).

For local command-line operations:

```powershell
.\scripts\windows\manage.ps1 status
.\scripts\windows\manage.ps1 restart
.\scripts\windows\manage.ps1 logs
```

`stop` stops the managed engine while leaving the tray/dashboard available. `start` launches the
installed app if needed, or starts its engine. Dashboard controls provide the same engine actions.

## Dashboard

The dashboard is built to stay open in a narrow window beside other work. A mode row at the top
of every page shows the engine state with its Start, Restart and Stop keys, the model, the running
plan and the API address (select it to copy). **Monitor** holds the live readings without
scrolling: GPU memory as a scale tape and history with the desktop reserve drawn as a red-striped
band, generation and prompt-reading rates on separate scales, the calling apps, and the cautions
from the findings. Pointing at either history, or using the arrow keys on it, reads both at the
same moment. In a wide window the gaps between the Monitor displays can be dragged, or moved with
the arrow keys, to give memory, rates or the calling apps more room; double-click or Enter restores
the default split, and the split is kept in the browser. Readings travel briefly to each new value
instead of jumping; with Windows animation effects turned off they change at once. The other pages
keep a one-line memory and rate band in view.

**Chat**, **Connect**, **API**, **Diagnose** (findings, engine log, engine data, acceleration, expert use) and **Settings**
are pages of one control display. The Settings index shows each page's current values; model presets
and the desktop reserve have their own pages there. Edits are staged as a temporary revision,
shown in amber, until **Insert** saves them or **Erase** discards them; a staged edit also shows
under the engine mode. **Explain** shows plain-language notes on every page. **Master caution**
lights for a caution or warning finding that has not been acknowledged in this browser; pressing
it acknowledges the current findings and opens Diagnose.
The display-mode key switches between **Auto**, which follows the Windows app theme, **Day** and
**Night**; the choice is kept in the browser.

**Chat** talks to the served model. Each reply shows how long it took to start, its prompt-reading
time and generation rate, how much of the conversation the engine reused instead of reading it
again, speculative-draft acceptance, why it finished, and a **Sent** line with exactly the
parameters the request set. The page imposes no output limit of its own: by default it leaves the
limit out, so the engine's `--default-max-tokens` applies (8192 unless the launch sets another
value), and **Parameters** sets a custom budget or no limit (`max_tokens` `-1`, which ends the
answer only at a stop condition or a full context). Parameters also hold the sampling and reasoning
fields the engine accepts, stop strings, a response format or JSON schema, token probabilities
(answers then arrive whole, and a token view marks the tokens the model was unsure of), read-only
prompt-cache participation for tests that should not evict other apps' cached conversations, and
extra request JSON merged into the request last. A field left empty sends nothing. Images can be
attached, pasted or dropped when the model reads them, and **Open in API** carries the next request
to the API page. The conversation lasts as long as the browser tab; the parameters are kept in the
browser.

**API** is a test bench for every route the engine exposes (status, OpenAI, Anthropic, NInfer and
admin) and for the Supervisor's own read routes. Each route has ready examples; the answer shows
its status and timings, the assembled text of a stream, its raw events, the body and the headers,
and **Copy as curl** writes the same request against the engine's own address. The page sends
engine requests through the Supervisor at `/api/engine/<engine route>`, which adds the configured
API key and streams each answer as the engine writes it. That route forwards engine routes only
and accepts only loopback callers that send the `X-NInfer-Supervisor` header, like the engine
controls. Unlike a web frontend it can call the engine's admin writes; the page asks before it
sends `POST /admin/quiesce`, which holds incoming requests while it runs.

**Diagnose › Experts** shows how busy the model's routed experts have been, for an engine started
with `--expert-stats` (or `--expert-trace` with a folder, which also writes a trace file). The map has one
row per text layer and one column per expert, with the speculative-decoding layer, when there is one,
set apart below. Brightness is each row's own scale, so a layer's busiest expert is always the
brightest; counts fade with a 60-second half-life, so the picture follows what the engine is doing
now. **Decode** (green, like the generation rate) covers answers, including drafts and their checks;
**Prefill** (blue, like the prompt-reading rate) covers prompts. **By expert** keeps every expert in
its own column, which shows which experts stay busy; **Sorted** lines each layer up from busiest to
quietest, so the slope shows how lopsided the use is. Pointing at the map, or using the arrow keys
on it, reads out the layer, expert, rank, heat, its share of the layer and the share held by the
experts up to it. The bars at the right give, per layer, how many experts it takes to cover 80% of
that layer's activity, and a line under the map gives the smallest, median and largest. The counters
line shows steps and tokens per kind, uptime, the half-life and the trace file (its size, what is
queued, and how many records were dropped, in amber when any were).

The **Expert VRAM budget** control asks how much of the recent decode activity the busiest experts
would cover. It takes a size in experts or GiB (one expert is 2,764,808 bytes; the text layers hold
24,576, 63.3 GiB) and reports two shares: the hottest experts across all layers, and an even split
with the same number in every layer. Both are upper bounds for a fixed set chosen from the traffic
on screen, not measured hit rates, and the speculative layer's experts are left out. In Sorted order
magenta marks show where each choice would cut. The page asks the engine's `/admin/experts` through the
Supervisor every two seconds while it is open and visible, and stops otherwise. Without the engine
option it explains how to turn it on. The Supervisor has no setting for it, so the flag goes in the
engine arguments of its configuration file, where flags it does not describe are passed through
unchanged; Settings › Launch details shows the arguments the engine runs with.

## Web frontends

The Supervisor can serve static web apps, such as llama.cpp's web UI, beside the engine. Each
frontend gets its own loopback port, and so its own browser origin, which also serves the engine's
API. A page there needs neither `--cors` nor the engine's API key, and several frontends can run
side by side without sharing browser storage.

Add, change or remove frontends in the dashboard under **Settings › Web frontends**: give each a name,
the folder of a built app and a port, then **Insert and apply**. The Supervisor saves the list to its
configuration and starts or stops the listeners right away, while the engine keeps running.
Removing or changing a frontend ends the chats open in it, and saving an unchanged entry that is
not serving retries it, for example after its files were copied in. The list is stored in the
configuration file as:

```json
"frontends": [
  {"name": "llama.cpp web UI", "dir": "C:/NInfer/frontends/llama-ui", "port": 8100}
]
```

`dir` holds a built app with `index.html` at its root. llama.cpp publishes prebuilt web UI assets on
Hugging Face (`ggml-org/llama-ui`), or build `tools/ui` from the llama.cpp sources. Open a frontend
from the tray's **Open** entry or from the dashboard's **Connect** page.

On a frontend port, the Supervisor:

- serves files from `dir`, with `/` answered by `index.html`;
- forwards `/v1/*`, `/props`, `/health`, `/systemone` and read-only `/admin/*` requests to the
  engine, streams each answer as the engine writes it, and cancels the engine request when the
  browser closes the stream;
- adds the API key from `engine.api_key_file` when one is configured, replacing any key the page
  sends;
- answers a page navigation to any other path with `index.html`, so client-side routes reload, and
  forwards every other request to the engine, which answers routes it does not have with 404.

Frontends listen on `127.0.0.1` only, because the proxy holds the engine's key. A request whose
`Host` is not the frontend's loopback address, or whose `Origin` is another site, is refused, and
engine admin writes such as `POST /admin/quiesce` are not available there. A hand edit of the
configuration file applies at the next Supervisor start. Ports must be distinct and differ from the
Supervisor and engine ports, with at most 16 frontends; a dashboard or engine port change that would
collide with a frontend is refused. A missing folder or a port already in use
is reported in the tray and the dashboard and does not stop the Supervisor.

## Request capacity

Settings → Request capacity sizes three related Engine startup values:

1. **KV capacity** is the shared token pool for every active request. Engine default omits the
   flag so the pool follows max context. Auto sizes the pool from free GPU memory at startup.
2. **Max context** is the longest one request may be. It cannot exceed an explicit pool; enlarge
   the pool first. Lowering the pool pulls max context down.
3. **Max concurrency** is how many requests may be Active at once. The control’s maximum of 8 is
   the Engine compile-time lane cap (`kMaximumConcurrency`): CUDA Graphs and kernels are built
   for exact batch `1..8`. It is not a GPU-memory estimate. Omitting it uses 1 lane.

An explicit pool must cover one full-length request and cannot exceed
`max concurrency × max context`. The live preview shows how many full-length requests the pool
can hold against the configured lane count. Extra lanes share the pool and must use shorter
requests. **Insert** does not restart the engine; use **Insert + restart**, or **Restart** beside the
engine mode, when the new capacity should take effect.

To update, rebuild Release and rerun `install.ps1` without `-ConfigPath`. The installer validates
runtime dependencies before stopping the installed app, replaces the binaries, retains one previous
binary directory, and relaunches through Explorer. Configuration, logs and tray preferences remain.
The login preference remains as selected in the tray. `-NoStart` installs without launching.

Uninstall through **Settings → Apps → Installed apps → NInfer**, or run the installed
`uninstall.ps1`. Uninstall stops the installed processes and removes the app, shortcuts and its
login entry. Configuration, logs and model artifacts are retained.
