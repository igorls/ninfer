' start-supervisor.vbs - launch the NInfer supervisor launcher with no visible console window.
' Invoked by the "NInfer Production Server" scheduled task at logon or on demand.
Set sh = CreateObject("WScript.Shell")
cmd = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File ""P:\NInfer\scripts\service\start-supervisor.ps1"" -Foreground"
sh.Run cmd, 0, False
