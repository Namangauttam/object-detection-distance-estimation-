# Configure and build yolo_infer.exe (Release) with MSYS2 UCRT64 g++ + Ninja.
#   powershell -ExecutionPolicy Bypass -File src\build.ps1
$ErrorActionPreference = "Stop"
$Src = $PSScriptRoot
$Build = Join-Path $Src "build"
$Msys = if ($env:MSYS2_ROOT) { $env:MSYS2_ROOT } else { "C:\msys64" }
$Ucrt = Join-Path $Msys "ucrt64"

$haveOrt = Test-Path (Join-Path $Src "..\third_party\onnxruntime\lib\onnxruntime.dll")
$haveOcv = (Get-ChildItem (Join-Path $Ucrt "lib\cmake") -Filter "opencv*" -ErrorAction SilentlyContinue) -ne $null
if (-not ($haveOrt -and $haveOcv)) { & (Join-Path $Src "setup_deps.ps1") }

# Find a Python that has PyTorch (its CUDA/cuDNN DLLs are used by ONNX Runtime). MSYS2 ships its
# own python without torch, so test each candidate instead of taking the first on PATH.
$python = $null
$candidates = @(Get-Command python, python3 -All -ErrorAction SilentlyContinue | ForEach-Object Source) +
              @(Get-ChildItem "$env:LOCALAPPDATA\Programs\Python\Python3*\python.exe" -ErrorAction SilentlyContinue | ForEach-Object FullName)
$ErrorActionPreference = "Continue"   # PS 5.1 turns native stderr into errors under "Stop"
foreach ($p in $candidates | Select-Object -Unique) {
    & $p -c "import torch" 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) { $python = $p; break }
}
$ErrorActionPreference = "Stop"
if (-not $python) { Write-Warning "No Python with PyTorch found; GPU inference will need CUDA_DLL_DIR set." }
$env:PATH = "$Ucrt\bin;$env:PATH"
$cmake = (Get-Command cmake -ErrorAction SilentlyContinue).Source
if (-not $cmake) { throw "CMake not found. Install it with: pip install cmake" }

& $cmake -S $Src -B $Build -G Ninja `
    -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_CXX_COMPILER="$Ucrt/bin/g++.exe" `
    -DCMAKE_MAKE_PROGRAM="$Ucrt/bin/ninja.exe" `
    -DCMAKE_PREFIX_PATH="$Ucrt" `
    -DPython3_EXECUTABLE="$python"
if ($LASTEXITCODE) { throw "CMake configure failed" }
& $cmake --build $Build --parallel
if ($LASTEXITCODE) { throw "Build failed" }
Write-Host "`nBuilt: $(Join-Path $Build 'yolo_infer.exe')"
