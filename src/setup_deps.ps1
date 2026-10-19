# One-time setup of the C++ dependencies (MSYS2 UCRT64 toolchain).
#   powershell -ExecutionPolicy Bypass -File src\setup_deps.ps1
#   powershell -ExecutionPolicy Bypass -File src\setup_deps.ps1 -TensorRT   # also TensorRT (2.2 GB download)
param([switch]$TensorRT)
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

# Optional: TensorRT 10 runtime DLLs for the ONNX Runtime TensorRT provider (~2x faster than CUDA).
# Extracted straight into third_party\tensorrt (3.2 GB) so nothing is installed on the system drive.
$TrtVersion = "10.16.1.11"
$TrtDir = Join-Path $Tp "tensorrt"
if ($TensorRT -and -not (Test-Path (Join-Path $TrtDir "nvinfer_10.dll"))) {
    $whl = Join-Path $Tp "tensorrt_cu12_libs-$TrtVersion-py3-none-win_amd64.whl"
    $url = "https://pypi.nvidia.com/tensorrt-cu12-libs/tensorrt_cu12_libs-$TrtVersion-py3-none-win_amd64.whl"
    Write-Host "Downloading TensorRT $TrtVersion (2.2 GB) ..."
    & curl.exe -L -C - --retry 5 -o $whl $url
    if ($LASTEXITCODE) { throw "TensorRT download failed" }
    New-Item -ItemType Directory -Force $TrtDir | Out-Null
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [System.IO.Compression.ZipFile]::OpenRead($whl)
    try {
        foreach ($e in $zip.Entries) {
            if ($e.FullName -like "tensorrt_libs/*.dll") {
                [System.IO.Compression.ZipFileExtensions]::ExtractToFile($e, (Join-Path $TrtDir $e.Name), $true)
            }
        }
    } finally { $zip.Dispose() }
    Remove-Item $whl
}

Write-Host "Dependencies ready."
