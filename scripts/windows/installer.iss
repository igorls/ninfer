#ifndef PayloadDir
  #error PayloadDir must point to the staged runtime
#endif
#ifndef ReleaseVersion
  #define ReleaseVersion "2026.09.11-preview.1"
#endif
#ifndef OutputDir
  #define OutputDir "..\..\dist"
#endif

[Setup]
AppId=NInfer
AppName=NInfer
AppVersion={#ReleaseVersion}
AppPublisher=NInfer workstation fork
AppPublisherURL=https://github.com/igorls/ninfer
AppSupportURL=https://github.com/igorls/ninfer/issues
DefaultDirName={localappdata}\Programs\NInfer
DefaultGroupName=NInfer
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.22000
DisableProgramGroupPage=yes
AllowNoIcons=yes
UninstallDisplayIcon={app}\bin\ninfer-launcher.exe
SetupIconFile=..\..\apps\windows\assets\ninfer.ico
LicenseFile={#PayloadDir}\LICENSE
InfoBeforeFile={#PayloadDir}\INSTALL-NOTES.txt
OutputDir={#OutputDir}
OutputBaseFilename=NInfer-{#ReleaseVersion}-windows-x64-unsigned
Compression=lzma2/normal
SolidCompression=yes
WizardStyle=modern
SignedUninstaller=no
CloseApplications=no
RestartApplications=no
UninstallLogMode=append
VersionInfoVersion=2026.9.11.1

[Files]
Source: "{#PayloadDir}\bin\*"; DestDir: "{app}\bin"; Flags: ignoreversion
Source: "{#PayloadDir}\licenses\*"; DestDir: "{app}\licenses"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#PayloadDir}\LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#PayloadDir}\INSTALL-NOTES.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#PayloadDir}\release-manifest.json"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\NInfer"; Filename: "{app}\bin\ninfer-launcher.exe"; WorkingDir: "{app}"; IconFilename: "{app}\bin\ninfer-launcher.exe"
Name: "{group}\NInfer documentation"; Filename: "https://github.com/igorls/ninfer#readme"
Name: "{group}\Uninstall NInfer"; Filename: "{uninstallexe}"

[Run]
Filename: "{app}\bin\ninfer-launcher.exe"; Description: "Open NInfer and choose a model"; Flags: nowait postinstall skipifsilent

[Code]
function AppIsRunning(): Boolean;
var
  Locator, Services, Processes, Process: Variant;
  I: Integer;
  Path: String;
begin
  Result := False;
  Locator := CreateOleObject('WbemScripting.SWbemLocator');
  Services := Locator.ConnectServer('', 'root\CIMV2');
  Processes := Services.ExecQuery('SELECT ExecutablePath FROM Win32_Process WHERE Name=''ninfer-supervisor.exe'' OR Name=''ninfer-serve.exe'' OR Name=''ninfer-launcher.exe''');
  for I := 0 to Processes.Count - 1 do begin
    Process := Processes.ItemIndex(I);
    if not VarIsNull(Process.ExecutablePath) then begin
      Path := Process.ExecutablePath;
      if (CompareText(Path, ExpandConstant('{app}\bin\ninfer-supervisor.exe')) = 0) or
         (CompareText(Path, ExpandConstant('{app}\bin\ninfer-serve.exe')) = 0) or
         (CompareText(Path, ExpandConstant('{app}\bin\ninfer-launcher.exe')) = 0) then begin
        Result := True;
        Exit;
      end;
    end;
  end;
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
begin
  Result := '';
  try
    if AppIsRunning() then
      Result := 'Quit NInfer from its tray menu, then retry. Your models and settings will be preserved.';
  except
    Result := 'Could not check whether NInfer is running. Close NInfer and retry.';
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  PreviousLocation: String;
begin
  if CurStep = ssPostInstall then begin
    { Retire the old source-build installer's entry only for this exact app directory. }
    if RegQueryStringValue(HKCU, 'Software\Microsoft\Windows\CurrentVersion\Uninstall\NInfer', 'InstallLocation', PreviousLocation) and
       (CompareText(RemoveBackslashUnlessRoot(PreviousLocation), RemoveBackslashUnlessRoot(ExpandConstant('{app}'))) = 0) then
      RegDeleteKeyIncludingSubkeys(HKCU, 'Software\Microsoft\Windows\CurrentVersion\Uninstall\NInfer');
  end;
end;

function InitializeUninstall(): Boolean;
begin
  Result := False;
  try
    if AppIsRunning() then begin
      MsgBox('Quit NInfer from its tray menu before uninstalling. Models and settings will be kept.', mbInformation, MB_OK);
      Exit;
    end;
    Result := True;
  except
    MsgBox('Could not check whether NInfer is running. Close NInfer and retry.', mbError, MB_OK);
  end;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  LoginCommand: String;
begin
  if CurUninstallStep = usUninstall then begin
    if RegQueryStringValue(HKCU, 'Software\Microsoft\Windows\CurrentVersion\Run', 'NInferSupervisor', LoginCommand) then
      if (Pos(Lowercase('"' + ExpandConstant('{app}\bin\ninfer-supervisor.exe') + '"'), Lowercase(LoginCommand)) = 1) or
         (Pos(Lowercase('"' + ExpandConstant('{app}\bin\ninfer-launcher.exe') + '"'), Lowercase(LoginCommand)) = 1) then
        RegDeleteValue(HKCU, 'Software\Microsoft\Windows\CurrentVersion\Run', 'NInferSupervisor');
  end;
end;
