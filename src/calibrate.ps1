# Lens calibration wrapper (adds the MSYS2 DLLs to PATH and runs from the project root).
#   powershell -ExecutionPolicy Bypass -File src\calibrate.ps1 --make-board board.png
#   powershell -ExecutionPolicy Bypass -File src\calibrate.ps1 --source calib.mp4 --square 0.025 --show
$Root = Split-Path -Parent $PSScriptRoot
$Msys = if ($env:MSYS2_ROOT) { $env:MSYS2_ROOT } else { "C:\msys64" }
$Exe = Join-Path $PSScriptRoot "build\calibrate_camera.exe"
if (-not (Test-Path $Exe)) { throw "Not built yet. Run src\build.ps1 first." }
$env:PATH = "$Msys\ucrt64\bin;$env:PATH"
Push-Location $Root
try { & $Exe @args } finally { Pop-Location }
