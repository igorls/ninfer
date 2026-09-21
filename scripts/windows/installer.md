# Windows installer

Build the engine and Supervisor from the intended source revision in an isolated
checkout, with `NINFER_BUILD_MEDIA=ON` if Vision is to be included. Build the
first-launch application separately:

```powershell
cmake -S scripts/windows/launcher -B build-launcher -G "Visual Studio 18 2026" -A x64
cmake --build build-launcher --config Release -j
```

Run `scripts/windows/build-installer.ps1` with explicit engine build, launcher
build, Inno Setup compiler, FFmpeg license, and curl license paths. The script
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

The current candidate is for internal unsigned testing. Before a public release,
provide corresponding source/build materials for the bundled multimedia libraries
and review all runtime redistribution notices. The workstation's existing FFmpeg
DLLs are a GPL build, not an LGPL build. The notes shipped in this candidate state
that limitation explicitly. Do not describe local installation checks as 5090
inference qualification until the receiving machine has run the model.
