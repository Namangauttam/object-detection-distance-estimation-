# One-time setup of the C++ dependencies (MSYS2 UCRT64 toolchain).
#   powershell -ExecutionPolicy Bypass -File src\setup_deps.ps1
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # makes Invoke-WebRequest much faster

$Msys = if ($env:MSYS2_ROOT) { $env:MSYS2_ROOT } else { "C:\msys64" }
$OrtVersion = "1.31.0"
$Root = Split-Path -Parent $PSScriptRoot
$Tp = Join-Path $Root "third_party"
New-Item -ItemType Directory -Force $Tp | Out-Null

# g++, OpenCV, Ninja, pkgconf from MSYS2
& "$Msys\usr\bin\bash.exe" -lc "pacman -S --needed --noconfirm mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-opencv mingw-w64-ucrt-x86_64-ninja mingw-w64-ucrt-x86_64-pkgconf"
if ($LASTEXITCODE) { throw "pacman install failed" }

# CMake (pip build works fine with the MinGW toolchain)
if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) { pip install cmake }

# ONNX Runtime (GPU, CUDA 12). CUDA/cuDNN runtime DLLs come from the PyTorch install.
$OrtDir = Join-Path $Tp "onnxruntime"
if (-not (Test-Path (Join-Path $OrtDir "lib\onnxruntime.dll"))) {
    $zip = Join-Path $Tp "ort.zip"
    $url = "https://github.com/microsoft/onnxruntime/releases/download/v$OrtVersion/onnxruntime-win-x64-gpu_cuda12-$OrtVersion.zip"
    Write-Host "Downloading ONNX Runtime $OrtVersion ..."
    Invoke-WebRequest $url -OutFile $zip
    Expand-Archive $zip -DestinationPath $Tp -Force
    if (Test-Path $OrtDir) { Remove-Item -Recurse -Force $OrtDir }
    Rename-Item (Join-Path $Tp "onnxruntime-win-x64-gpu_cuda12-$OrtVersion") "onnxruntime"
    Remove-Item $zip
}

Write-Host "Dependencies ready."
