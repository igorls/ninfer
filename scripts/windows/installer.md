# Windows installer

Build the engine and Supervisor from the intended source revision in an isolated
checkout. Engine builds include the media libraries; Vision is enabled at runtime.
Build the first-launch application separately:

```powershell
cmake -S scripts/windows/launcher -B build-launcher -G "Visual Studio 18 2026" -A x64
cmake --build build-launcher --config Release -j
```

Run `scripts/windows/build-installer.ps1` with explicit engine build, launcher
build, Inno Setup compiler, FFmpeg license, FFmpeg source note, and curl license paths. The script
requires an explicit `-Version yyyy.MM.dd-channel.revision` and derives the installer's numeric
Windows file version from the same date and revision. The hwloc notice staged with the runtime
is installed under `licenses/hwloc/`; the libjpeg-turbo license and IJG acknowledgment
are staged under `licenses/libjpeg-turbo/`. Engine configuration requires `JPEG_ROOT`
for the Windows SDK, and packaging resolves its runtime DLL from that SDK's `bin/`.
The script
stages runtime DLLs, checks all four executables without development tools on
PATH, builds the unsigned Setup executable, and writes its SHA-256 checksum.
Use a new version for each candidate; a staging directory is never overwritten.

The installer needs no compiler or toolkit on the receiving computer. NInfer's
launcher checks GPU architecture and driver availability, opens a native model
picker on first launch, and creates a loopback-only configuration. It does not
download model weights or overwrite an existing configuration. CLI provisioning
uses the same checks with `--model FILE --no-start`; `--data-dir DIR`,
`--engine-port PORT` and `--dashboard-port PORT` allow isolated validation.
`--probe` checks hardware without loading a model.

Setup reads the v3 entry header and directory for the optional display name and
MTP component. It does not maintain a checkpoint registry or validate model
mathematics; the Engine validates and admits the complete artifact at load.
Artifacts without MTP use ordinary decoding. The initial profile keeps the
Engine's 8 GiB desktop reserve.
Qwen3_5MTP starts with its optimized proposal head; Flash-Next starts with its
required full shared head. The Flash setup link opens the compatible-artifact
guide because the published legacy entry requires an offline reframe.

Install, upgrade and uninstall checks must use a separate `/DIR` and `/NOICONS`
on a development machine; do not launch the installed app against production
configuration during these checks. Model and settings preservation, bundled
runtime startup and actual inference on the receiving GPU are acceptance checks.
Inno uses the same per-user uninstall registry entry for every installation
directory. A separate `/DIR` does not isolate that entry: use a disposable user
environment or save and restore the exact existing NInfer entry during testing.

Build Vision against an LGPL shared FFmpeg (for example BtbN's `win64-lgpl-shared`
builds), never a GPL build: NInfer only decodes media, and the LGPL decoders cover it.
The `-FFmpegSourceNote` file names the exact FFmpeg version and where its corresponding
source is published; ship that source with every public release (attach the source
archive for the same commit to the GitHub release). The DLLs stay dynamically linked so
users can replace them. Do not describe local installation checks as 5090 inference
qualification until the receiving machine has run the model.
