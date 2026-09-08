# Windows app

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
| Active configuration | `%LOCALAPPDATA%\NInfer\supervisor.json` |
| Tray preferences | `%LOCALAPPDATA%\NInfer\supervisor.tray.json` |
| Engine and request logs | `%LOCALAPPDATA%\NInfer\logs` |
| Application shortcuts | Start menu → NInfer |

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

To update, rebuild Release and rerun `install.ps1` without `-ConfigPath`. The installer validates
runtime dependencies before stopping the installed app, replaces the binaries, retains one previous
binary directory, and relaunches through Explorer. Configuration, logs and tray preferences remain.
The login preference remains as selected in the tray. `-NoStart` installs without launching.

Uninstall through **Settings → Apps → Installed apps → NInfer**, or run the installed
`uninstall.ps1`. Uninstall stops the installed processes and removes the app, shortcuts and its
login entry. Configuration, logs and model artifacts are retained.
