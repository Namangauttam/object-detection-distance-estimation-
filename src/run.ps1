# Run live inference on every video in ..\videos and save annotated copies to ..\inference_output.
#   powershell -ExecutionPolicy Bypass -File src\run.ps1
#   powershell -ExecutionPolicy Bypass -File src\run.ps1 --conf 0.35 --source videos\cones_edited_4x.mp4
#   powershell -ExecutionPolicy Bypass -File src\run.ps1 --no-show
$Root = Split-Path -Parent $PSScriptRoot
$Msys = if ($env:MSYS2_ROOT) { $env:MSYS2_ROOT } else { "C:\msys64" }
$Exe = Join-Path $PSScriptRoot "build\yolo_infer.exe"
if (-not (Test-Path $Exe)) { throw "Not built yet. Run src\build.ps1 first." }
$env:PATH = "$Msys\ucrt64\bin;$env:PATH"   # OpenCV + MinGW runtime DLLs
Push-Location $Root
try { & $Exe @args } finally { Pop-Location }
