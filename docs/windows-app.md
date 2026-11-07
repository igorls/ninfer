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
| Application shortcuts | Start menu → NInfer Workstation |

The dashboard's request figures and its Diagnose findings come from the request log. The
log is never trimmed, so the Supervisor reads an existing one once, in the background, when it
starts (about 8 s for 750 MiB) and the findings show that progress until it finishes; after that
it reads only the records the engine appends.

**Calling apps** on Monitor reports observed requests over a rolling 15-minute window, grouped by
User-Agent. Completed, cancelled, failed, unfinished-at-restart and awaiting-outcome requests are separate; admission
timeouts are counted explicitly. First-token, queue and prefill timings include means,
p95 values; an app whose queue wait averages a second or more is shaded and marked **Long wait**, and
requests that expired before admission add a line of their own. Queue timing is labelled as first-output or
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

The dashboard is built to stay open in a narrow window beside other work. Every page shares one
look: a neutral scale in Day and Night, sentence-case lettering, a single violet accent for focus,
unsaved changes and the button that commits them, and red only for something that fails. Under the
page tabs, an engine line on every page shows the engine state with its Start, Restart and Stop
buttons, the model, the running plan when the launch arguments are known (context pool, max context,
requests at once, KV format and speculation) and the API address (select it to copy). **Monitor** holds the live readings without
scrolling: GPU memory as a scale and history with the desktop reserve drawn as a hatched band above a
dashed line (red while the engine is inside it), generation and prompt-reading rates on separate
scales, the calling apps, and the alerts from the findings. Pointing at either history, or using the arrow keys on it, reads both at the
same moment. In a wide window the gaps between the Monitor displays can be dragged, or moved with
the arrow keys, to give memory, rates or the calling apps more room; double-click or Enter restores
the default split, and the split is kept in the browser. Readings travel briefly to each new value
instead of jumping; with Windows animation effects turned off they change at once. Chat, Connect,
API and Diagnose keep a one-line memory and rate band in view.

**Chat**, **Connect**, **API** and **Diagnose** (findings, engine log, engine data, acceleration,
expert use) each fill the page below the engine line. **Settings** is one page holding every setting, in
sections for Model, Memory, Requests, Access, Web frontends, and Logs and launch. The index beside it
shows each section's current values and how many changes are unsaved. The search field finds a
setting by any word of its name or help, by its `--flag`, or by its current value; `/` jumps to it.
Changes stay on the page, marked in violet, until **Save** writes them or **Discard** drops them;
**Save and restart** also restarts the engine, and unsaved changes also show in the engine line.
**Explain** shows plain-language notes on every page. The alerts button counts findings that have not
been acknowledged in this browser and shows **No alerts** otherwise; pressing it acknowledges the
current findings and opens Diagnose.
The display-mode button switches between **Auto**, which follows the Windows app theme, **Day** and
**Night**; the choice is kept in the browser.

**Chat** talks to the served model across a full-width two-pane workspace: the conversation on the
left and a permanent **Reply inspector** + **Parameters** sidebar on the right (which collapses behind
the **Parameters** button in narrow windows). Each reply shows a compact vitals strip with its generation
rate, first-token or whole-response time, output token count, prompt-cache reuse, speculative-draft
acceptance, token confidence and any non-normal finish status, alongside **Inspect** (which selects
that turn in the **Reply inspector** to show its full telemetry breakdown and the **Sent** line with
the exact parameters the request set) and the reply's actions (**Copy**, **Token view**,
**Regenerate**). User prompts also carry a **Copy** button. Replies, and the reasoning above them, are formatted as Markdown while they arrive: headings,
lists, task lists, tables, quotes, code blocks with their language and a **Copy** button that copies
the code as written, and links. Only http, https and mailto addresses become links (opened in a new
tab); images are never fetched, and HTML in a reply is shown as text. The page imposes no output
limit of its own: by default it sets no limit (`max_tokens` `-1`, which ends the answer only at a
stop condition or a full context), while **Parameters** can switch to the engine's
configured default (`generation.max_tokens` or `--max-tokens`, 8192 unless the launch
sets another value) or a custom token budget.
Parameters also hold the sampling and reasoning fields the engine accepts, stop strings, a response
format or JSON schema, token probabilities (answers then arrive whole, and a token view marks
unsure tokens while an instant readout bar inspects each token's probability and top alternatives),
read-only prompt-cache participation for tests that should not evict other apps' cached
conversations, and extra request JSON merged into the request last. A field left empty sends
nothing. Images can be attached, pasted or dropped when the model reads them, and **Open in API**
carries the next request to the API page. The conversation lasts as long as the browser tab; the
parameters are kept in the browser.

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
set apart below. Shade is each row's own scale, so a layer's busiest expert is always the
strongest; counts fade with a 60-second half-life, so the picture follows what the engine is doing
now. **Decode** covers answers, including drafts and their checks; **Prefill** covers prompts. **By expert** keeps every expert in
its own column, which shows which experts stay busy; **Sorted** lines each layer up from busiest to
quietest, so the slope shows how lopsided the use is. Pointing at the map, or using the arrow keys
on it, reads out the layer, expert, rank, heat, its share of the layer and the share held by the
experts up to it. The bars at the right give, per layer, how many experts it takes to cover 80% of
that layer's activity, and a line under the map gives the smallest, median and largest. The counters
line shows steps and tokens per kind, uptime, the half-life and the trace file (its size, what is
queued, and how many records were dropped, in red when any were).

The **Expert VRAM budget** control asks how much of the recent decode activity the busiest experts
would cover. It takes a size in experts or GiB (one expert is 2,764,808 bytes; the text layers hold
24,576, 63.3 GiB) and reports two shares: the hottest experts across all layers, and an even split
with the same number in every layer. Both are upper bounds for a fixed set chosen from the traffic
on screen, not measured hit rates, and the speculative layer's experts are left out. In Sorted order
violet marks show where each choice would cut. The page asks the engine's `/admin/experts` through the
Supervisor every two seconds while it is open and visible, and stops otherwise. Without the engine
option it explains how to turn it on. The Supervisor has no setting for it, so the flag goes in the
engine arguments of its configuration file, where flags it does not describe are passed through
unchanged; Settings › Logs and launch shows the arguments the engine runs with.

## Web frontends

The Supervisor can serve static web apps, such as llama.cpp's web UI, beside the engine. Each
frontend gets its own loopback port, and so its own browser origin, which also serves the engine's
API. A page there needs neither `--cors` nor the engine's API key, and several frontends can run
side by side without sharing browser storage.

Add, change or remove frontends in the dashboard under **Settings › Web frontends**: give each a name,
the folder of a built app and a port, then **Save**. The Supervisor saves the list to its
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

## Memory settings

Settings › Memory decides how the GPU is split when the engine starts. A bar draws the whole card
at the next start: model weights, engine working memory, the context pool, other apps and what
stays free, with the desktop reserve marked inside the free part. Beside each control a panel shows
what the choice buys:

- **Context capacity** (`--context-capacity`) is the GPU memory every request shares for its KV
  cache and state. **Auto** takes what the desktop reserve and the engine's startup headroom leave.
  **Fixed size** takes a slider in whole GiB, or an exact quantity such as `20GiB` or `20480MiB`; a
  bare number means GiB. The slider runs from what this plan needs to what fits now and marks the
  running size. The panel shows how many full-length requests fit and how many tokens the pool holds.
- **Max context** (`--max-context`) is the longest prompt plus reply one request may use, and
  **KV format** (`--kv-dtype`) sets what each token costs. Its ceiling is the model's own position
  limit, read from the artifact and multiplied by the YaRN choice, not a fixed number.
- **Long context (YaRN)** (`--yarn-factor`) stretches a Qwen3.5-family model past its native
  length: 2 for 512K tokens, 4 for about 1M. It changes every request and may slightly lower
  quality on short texts, so leave it off unless you need the length. The Supervisor refuses
  DFlash2 beyond the draft's own 256K positions and offers no YaRN for Flash-Next. The running plan
  shows the Engine-reported position limit.
- **Concurrent requests** (`--max-concurrency`) is how many requests run at once, sharing the pool.
  Its maximum of 8 is the Engine compile-time lane cap (`kMaximumConcurrency`): CUDA Graphs and
  kernels are built for exact batch `1..8`, so it is not a memory estimate. The panel shows how long
  each request can be when all of them run.
- **Desktop reserve** (`--desktop-reserve`) is the GPU memory kept free at startup for Windows and
  other apps. Its limit follows the context capacity saved with it, so a smaller pool and a larger
  reserve can be saved together.

The figures come from the running engine. `/admin/vram` gives the plan, the startup budget and the
pool's block size, the Supervisor's own memory sample gives free memory, and the request log's
`server_start` record gives the Engine's token estimate for the pool. Free memory is a live reading,
so the next start differs if other apps change what they hold. When max context, concurrent
requests, prefix reuse, the prefill chunk or vision differ from the running plan, the minimum and the
token figures are modelled from the running plan's working sets and marked approximate until a
restart. A different KV cache format or speculative backend leaves the token figures blank until a
restart, and without a ready engine or a request log the panels say what is missing. A size below
this plan's exact minimum cannot be saved; a size that does not fit beside other apps is shown in red
but can be saved, because they may close before the next start. **Save** does not restart the engine;
use **Save and restart**, or **Restart** in the engine line, when the change should take effect.

To update, rebuild Release and rerun `install.ps1` without `-ConfigPath`. The installer validates
runtime dependencies before stopping the installed app, replaces the binaries, retains one previous
binary directory, and relaunches through Explorer. Configuration, logs and tray preferences remain.
The login preference remains as selected in the tray. `-NoStart` installs without launching.

The first managed start after an update replaces retired flags in `engine.args` and
`models[].args`, using the [startup migration table](startup-configuration.md#existing-fork-launches).
A token `--kv-capacity` becomes `--context-capacity auto`. The previous file stays beside the
configuration as `supervisor.json.pre-migration`, and the engine log lists every change. A
monitor-only Supervisor never rewrites the file. Flags the table does not know stay in place, and
the engine reports them when it refuses to start.

Uninstall through **Settings → Apps → Installed apps → NInfer**, or run the installed
`uninstall.ps1`. Uninstall stops the installed processes and removes the app, shortcuts and its
login entry. Configuration, logs and model artifacts are retained.
