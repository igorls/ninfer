# Windows installer

Build the engine and Supervisor from the intended source revision in an isolated
checkout, with `NINFER_BUILD_MEDIA=ON` if Vision is to be included. Build the
first-launch application separately:

```powershell
cmake -S scripts/windows/launcher -B build-launcher -G "Visual Studio 18 2026" -A x64
cmake --build build-launcher --config Release -j
```

Run `scripts/windows/build-installer.ps1` with explicit engine build, launcher
build, Inno Setup compiler, FFmpeg license, FFmpeg source note, and curl license paths. The script
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

Install, upgrade and uninstall checks must use a separate `/DIR` and `/NOICONS`
on a development machine; do not launch the installed app against production
configuration during these checks. Model and settings preservation, bundled
runtime startup and actual inference on the receiving GPU are acceptance checks.

Build Vision against an LGPL shared FFmpeg (for example BtbN's `win64-lgpl-shared`
builds), never a GPL build: NInfer only decodes media, and the LGPL decoders cover it.
The `-FFmpegSourceNote` file names the exact FFmpeg version and where its corresponding
source is published; ship that source with every public release (attach the source
archive for the same commit to the GitHub release). The DLLs stay dynamically linked so
users can replace them. Do not describe local installation checks as 5090 inference
qualification until the receiving machine has run the model.
