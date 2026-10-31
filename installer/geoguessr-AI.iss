; The Windows installer. GitHub builds it (see .github/workflows/setup.yml), with uv.exe beside this
; file in uv\ and a Python in python\. It needs no administrator: everything goes in the user's own
; programs folder, and that folder is where the bot works, keeping its model and rounds there too.
;
; It brings its Python rather than having uv fetch one, as uv would link to that with a junction, and
; what Setup starts can't follow one: Windows' redirection guard, which Setup turns on, stops it.

#ifndef Version
  #define Version "0.0.0"
#endif
#define Name "geoguessr-AI"

[Setup]
AppId={{7E1C2B74-4C0E-4B6B-9C46-2A7F3C5D9E10}
AppName={#Name}
AppVersion={#Version}
AppVerName={#Name} {#Version}
AppPublisherURL=https://github.com/andrey-rublev/geoguessr-AI
DefaultDirName={localappdata}\Programs\{#Name}
DisableDirPage=yes
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=Output
OutputBaseFilename={#Name}-Setup
SetupIconFile=..\src\geoguessr_ai\desktop.ico
UninstallDisplayIcon={app}\src\geoguessr_ai\desktop.ico
UninstallDisplayName={#Name}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.17763

[Messages]
WelcomeLabel2=This installs geoguessr-AI, an AI that plays OpenGuessr from what it sees on screen.%n%nIt fetches the libraries it needs as it installs, about 1 GB, which takes a few minutes. Then the app offers to download the model, about 2.2 GB.%n%nOnly play it against friends who know it's a bot, never in ranked or public games.

[Tasks]
Name: "desktopicon"; Description: "Put an icon on my desktop"

[InstallDelete]
; What the last version had and this one may not.
Type: filesandordirs; Name: "{app}\src"
Type: filesandordirs; Name: "{app}\scripts"

[Files]
Source: "..\src\*"; DestDir: "{app}\src"; Excludes: "__pycache__,*.pyc,*.egg-info"; Flags: recursesubdirs ignoreversion
Source: "..\scripts\wsl_setup.sh"; DestDir: "{app}\scripts"; Flags: ignoreversion
Source: "..\scripts\wsl_screen.sh"; DestDir: "{app}\scripts"; Flags: ignoreversion
Source: "..\scripts\wsl-constraints.txt"; DestDir: "{app}\scripts"; Flags: ignoreversion
Source: "..\pyproject.toml"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\uv.lock"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "install_libraries.cmd"; DestDir: "{app}\tools"; Flags: ignoreversion
Source: "uv\uv.exe"; DestDir: "{app}\tools"; Flags: ignoreversion
Source: "python\*"; DestDir: "{app}\python"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{autoprograms}\{#Name}"; Filename: "{app}\.venv\Scripts\pythonw.exe"; Parameters: "-m geoguessr_ai.desktop"; WorkingDir: "{app}"; IconFilename: "{app}\src\geoguessr_ai\desktop.ico"; Comment: "Start the geoguessr-AI bot and watch it play"
Name: "{autodesktop}\{#Name}"; Filename: "{app}\.venv\Scripts\pythonw.exe"; Parameters: "-m geoguessr_ai.desktop"; WorkingDir: "{app}"; IconFilename: "{app}\src\geoguessr_ai\desktop.ico"; Comment: "Start the geoguessr-AI bot and watch it play"; Tasks: desktopicon

[Run]
; By way of Explorer, like pressing its icon: started by Setup itself, it would be under the guard too.
Filename: "{win}\explorer.exe"; Parameters: """{autoprograms}\{#Name}.lnk"""; Description: "Open geoguessr-AI"; Flags: postinstall nowait skipifsilent; Check: LibrariesInstalled

[UninstallDelete]
; All it fetched or made since. Rounds it played stay in runs, for whoever wants them.
Type: filesandordirs; Name: "{app}\.venv"
Type: filesandordirs; Name: "{app}\python"
Type: filesandordirs; Name: "{app}\tools"
Type: filesandordirs; Name: "{app}\src"
Type: filesandordirs; Name: "{app}\models"
Type: filesandordirs; Name: "{app}\data"
Type: filesandordirs; Name: "{app}\.browser-profile"
Type: files; Name: "{app}\layout*"
Type: files; Name: "{app}\calibration_preview*"

[Code]
var
  Installed: Boolean;
  LastSaid: String;

function LibrariesInstalled: Boolean;
begin
  Result := Installed;
end;

{ Each line uv says, shown as it comes, so a long wait doesn't look like a hang. }
procedure Said(const S: String; const Error, FirstLine: Boolean);
begin
  if Trim(S) <> '' then begin
    LastSaid := Trim(S);
    WizardForm.FilenameLabel.Caption := LastSaid;
  end;
end;

procedure InstallLibraries;
var
  Code: Integer;
begin
  WizardForm.StatusLabel.Caption := 'Fetching the libraries it needs, about 1 GB. This takes a few minutes...';
  WizardForm.FilenameLabel.Caption := '';
  WizardForm.ProgressGauge.Style := npbstMarquee;
  try
    Installed := ExecAndLogOutput(ExpandConstant('{app}\tools\install_libraries.cmd'), '', ExpandConstant('{app}'), SW_HIDE, ewWaitUntilTerminated, Code, @Said) and (Code = 0);
  except
    Installed := False;
    LastSaid := GetExceptionMessage;
  end;
  WizardForm.ProgressGauge.Style := npbstNormal;
  if not Installed then
    SuppressibleMsgBox('The libraries it needs couldn''t be installed:'#13#10#13#10 + LastSaid + #13#10#13#10'Check your connection and that the disk has room, then run Setup again.', mbError, MB_OK, IDOK);
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
    InstallLibraries;
end;
