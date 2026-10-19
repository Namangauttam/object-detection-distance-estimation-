# C++ video inference (g++ / MSYS2)

`yolo_infer.exe` runs the fine-tuned YOLOv8s model on the videos using ONNX Runtime (TensorRT or CUDA) and OpenCV. It shows the detections live in a window and saves annotated copies to `inference_output/<video>_inference.mp4`.

## One-time setup

```powershell
powershell -ExecutionPolicy Bypass -File src\setup_deps.ps1 -TensorRT   # MSYS2 packages + ONNX Runtime GPU + TensorRT
```

This installs into MSYS2 UCRT64 (`C:\msys64`): g++, OpenCV, Qt6 (used by OpenCV's window), Ninja and pkgconf. It also downloads ONNX Runtime GPU into `third_party/onnxruntime`. The CUDA 12 and cuDNN 9 DLLs come from your PyTorch install, so no separate CUDA install is needed.

`-TensorRT` also downloads TensorRT 10.16 (2.2 GB) and extracts its DLLs (3.2 GB) into `third_party/tensorrt`. They go on the project drive, not the system drive. TensorRT is optional: without it the app uses CUDA.

## TensorRT

With TensorRT installed, `--provider auto` (the default) uses ONNX Runtime's TensorRT provider:

- It builds an **FP16 engine** from `best.onnx`. This takes about a minute and happens once.
- The engine is cached in `model_training/weights/trt_cache/` and loaded in a few seconds on later runs.
- Any layer TensorRT can't handle runs on CUDA instead.
- If TensorRT fails to load, the app falls back to CUDA, then to the CPU.

The engine is tied to the GPU, the driver, the TensorRT version and the model. **Delete `trt_cache/` after re-exporting the model** so it gets rebuilt.

```powershell
powershell -ExecutionPolicy Bypass -File src\run.ps1 --provider tensorrt   # force TensorRT
powershell -ExecutionPolicy Bypass -File src\run.ps1 --provider cuda       # CUDA only (uses best_fp16.onnx)
```

## Export the model (after each training run)

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe" src\export_onnx.py
```

This writes `model_training/weights/best.onnx` (FP32) and `best_fp16.onnx` (FP16). On the GPU, FP16 is used automatically.

## Build and run

```powershell
powershell -ExecutionPolicy Bypass -File src\build.ps1
powershell -ExecutionPolicy Bypass -File src\run.ps1                       # all videos, live window and saved copies
powershell -ExecutionPolicy Bypass -File src\run.ps1 --source videos\cones_edited_4x.mp4 --conf 0.35
powershell -ExecutionPolicy Bypass -File src\run.ps1 --no-show             # save only
powershell -ExecutionPolicy Bypass -File src\run.ps1 --provider cpu      # CPU fallback (about 210-270 ms per frame)
```

Options: `--model PATH`, `--source DIR|FILE`, `--out DIR`, `--conf 0.25`, `--iou 0.45`, `--camera src/camera.yaml`, `--provider auto|tensorrt|cuda|cpu` (`--cpu` is short for `--provider cpu`), `--debug-lanes`, `--no-show`.

Window keys: **Space** pauses, **N** skips to the next video, **Q** or **Esc** quits.

## Lens calibration

The tool needs a short video (or a folder of videos or photos) of a printed checkerboard, filmed with your camera.

```powershell
# 1. Make a board, print it at 100% scale, tape it flat to something rigid
powershell -ExecutionPolicy Bypass -File src\calibrate.ps1 --make-board board.png

# 2. Film it with the SAME camera, resolution and zoom as the driving videos (30-60 s).
#    Cover the whole frame, especially the corners and edges; tilt it; vary the distance.

# 3. Calibrate (measure one square: 25 mm = 0.025)
powershell -ExecutionPolicy Bypass -File src\calibrate.ps1 --source calib.mp4 --square 0.025 --show
```

This writes `fx, fy, cx, cy, dist, calib_width, calib_height` into `src/camera.yaml` and keeps every other setting. It also saves `src/calibration_preview.jpg`, showing the original frame beside the lens-corrected one. Aim for an RMS error below 0.5 px and image coverage of at least 14/16. `--source` also accepts a folder of photos. If you use a different board, pass its inner-corner count with `--board 9x6`.

The tool was validated on a synthetic video rendered with a known lens (fx = 500, k1 = −0.30). It recovered fx = 499.1 and k1 = −0.298, with 0.28 px RMS.

## Distance to objects

Every detection is labelled with its estimated distance from the camera, for example `cone 0.89 | 8.3 m`. A `~` prefix means the object's bottom is hidden by the bonnet, so only the size-based estimate was used. The values are written per detection to `inference_output/<video>_distances.csv`.

How it works ([distance.cpp](distance.cpp)):

1. **Ground plane.** The bottom-centre of the box touches the road. Given the camera height and the pitch, the ray through that pixel meets a flat road at one distance.
2. **Known size.** `distance = focal_px × real_height / box_height_px`, using the per-class heights in [camera.yaml](camera.yaml).
3. **Fusion.** The two estimates are combined, each weighted by its expected error. The ground-plane estimate is trusted more up close; the size estimate takes over when the bottom of the object is hidden.
4. **Dynamic horizon.** Each frame, lane lines are found (Canny + Hough) and the point where they meet gives the horizon row. This follows pitch changes as the car bounces, and the horizon is drawn as a dashed line (cyan = tracking, orange = searching). Use `--debug-lanes` to see the detected lane lines.
5. **Auto camera height.** Objects of a standard size (cones) give both estimates, and the camera height that makes them agree is learned as a running median.
6. **Tracking.** Detections are linked across frames by box overlap, and each object's distance is smoothed with a 1-D Kalman filter.

**What the system cannot learn by itself: the focal length.** Both estimates scale linearly with it, so a wrong field of view makes every distance proportionally wrong. The default `hfov_deg: 110` is an assumption. To get real metres:

1. Calibrate the camera with a checkerboard, then put `fx, fy, cx, cy, dist` into `camera.yaml`. Alternatively, use the lens field of view from the camera's spec sheet.
2. Measure the real heights of your cones, chairs, houses and barriers.
3. Measure the camera height, or keep `auto_camera_height: 1` once the cone height is correct.
4. Validate: place cones at tape-measured distances (5, 10, 20, 30 m), record a clip, and compare the CSV to the tape.

Expect the error to grow with distance, because a fixed pixel error covers more metres further away. Slopes and humps break the flat-road assumption. If the videos come from different cars or mounts, give each its own `camera.yaml` via `--camera`.

## Performance (RTX 5050 laptop, 1280 px input, `house` clip, measured back to back)

| Stage | TensorRT FP16 | CUDA FP16 |
|---|---|---|
| Pre-process (letterbox, CHW) | 2.4 ms | 1.8 ms |
| Model | **7.2 ms** | 15.8 ms |
| Post-process (decode, NMS) | 0.1 ms | 0.2 ms |
| **Total** | **9.7 ms (about 100 FPS)** | 17.7 ms (about 56 FPS) |
| Start-up | about 7 s (cached engine); about 60 s first time | about 9 s |

The detection counts from the two providers match within FP16 rounding (for example 291 vs 289 cones).

## Files

- `main.cpp`: the inference app
- `distance.hpp`, `distance.cpp`: distance estimation (horizon, ground plane, known size, fusion, tracking)
- `camera.yaml`: camera and object-size settings for distance estimation
- `calibrate.cpp`, `calibrate.ps1`: the lens calibration tool (`calibrate_camera.exe`)
- `CMakeLists.txt`: the build configuration
- `setup_deps.ps1`, `build.ps1`, `run.ps1`: setup, build and run scripts
- `export_onnx.py`: exports the model to ONNX
