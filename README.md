# Robotics Object Detection: front-camera obstacles and distance

This project detects road obstacles (cones, people, chairs, model houses and barriers) in front-facing vehicle camera video and estimates each object's distance from the camera. It runs in real time on a laptop GPU.

It covers the whole pipeline:

1. **Annotation:** frames are extracted from video, pre-labelled with an open-vocabulary model (YOLO-World), then corrected by hand in a local web tool.
2. **Training:** YOLOv8s is fine-tuned on the reviewed labels.
3. **Inference (C++):** ONNX Runtime (CUDA) and OpenCV draw live detections, record the annotated video and log distances.
4. **Distance:** monocular estimation from ground-plane geometry and known object size, with a dynamic horizon, automatic camera-height estimation and per-object tracking.
5. **Lens calibration (C++):** checkerboard calibration that writes the camera's focal length and lens distortion into the config.

```
videos/*.mp4
   │  annotation/extract_frames.py
   ▼
frames ──► annotation/auto_label.py (YOLO-World) ──► web review tool ──► YOLO labels
                                                                            │
                                                     model_training/train.py (YOLOv8s)
                                                                            │
                                                     src/export_onnx.py ──► best_fp16.onnx
                                                                            │
camera.yaml ◄── src/calibrate.ps1 (checkerboard)                            ▼
     └────────────────────────────────────────────► src/yolo_infer.exe (C++, CUDA)
                                                     live window + annotated video + distance CSV
```

---

## Repository layout

```
annotation/          Frame extraction, YOLO-World pre-labelling, web review tool
  extract_frames.py    videos -> dataset/images (every 4th frame)
  auto_label.py        YOLO-World -> dataset/labels (YOLO format, never overwrites your edits)
  server.py            review tool at http://localhost:8765 (standard library only)
  tool/index.html      review UI: draw, delete, re-class boxes, add classes
  split_dataset.py     train/val lists
  dataset/             labels/, classes.txt, data.yaml (images/ are not in git)
model_training/      YOLOv8s fine-tuning
  train.py             builds a per-video train/val split and trains
  predict_video.py     quick Python check of a trained model
src/                 C++ real-time inference, distance estimation, lens calibration
  main.cpp             yolo_infer: ONNX Runtime + OpenCV, live window, recording, CSV
  distance.hpp/.cpp    horizon tracking, ground-plane and size distance, fusion, tracker
  calibrate.cpp        calibrate_camera: checkerboard -> fx, fy, cx, cy, distortion
  camera.yaml          camera and object-size settings
  export_onnx.py       .pt -> best.onnx (FP32) and best_fp16.onnx
  setup_deps.ps1 / build.ps1 / run.ps1 / calibrate.ps1
```

Not in git (see `.gitignore`): videos, extracted frames, inference outputs, model weights, training runs, `third_party/` and build output.

---

## Requirements

- Windows 10/11 with an NVIDIA GPU. Developed on an RTX 5050 Laptop GPU with 8 GB.
- **Python 3.11** with `torch` (CUDA build), `ultralytics` and `opencv-python`. This is used for annotation, training and ONNX export.
- **MSYS2** at `C:\msys64`, used for the C++ build (UCRT64 g++). `src\setup_deps.ps1` installs OpenCV, Qt6, Ninja and pkgconf there, and downloads ONNX Runtime GPU 1.31 into `third_party/`.
- CMake (`pip install cmake`).

> **PATH note:** installing OpenCV in MSYS2 also installs MSYS2's own Python. If `C:\msys64\ucrt64\bin` is on your *system* PATH, `python` may then launch MSYS2's Python, which has no torch. In that case, remove it from the system PATH or call your Python by its full path. The `.ps1` scripts add the MSYS2 folder themselves.

---

## Quick start

All commands run from the repository root.

### 1. Annotate

```powershell
cd annotation
python extract_frames.py        # videos/*.mp4 -> dataset/images
python auto_label.py            # YOLO-World pre-labels -> dataset/labels
python server.py                # open http://localhost:8765
```

In the review tool:
- click a box and press `Del` to remove a false detection;
- press `1`–`9`, then drag, to draw a missed object;
- use **Add** to create a new class;
- press `Enter` to mark the image reviewed and move on.

Full shortcuts are in [annotation/README.md](annotation/README.md).

### 2. Train

```powershell
cd model_training
python train.py                 # reviewed images only
python train.py --all           # include unreviewed auto-labels
```

The best weights are saved to `model_training/weights/best.pt`. On Windows the defaults are `--workers 0` and a disk image cache, to avoid `WinError 1455`. See [model_training/README.md](model_training/README.md).

### 3. Export and build the C++ app

```powershell
python src\export_onnx.py                                        # best.onnx + best_fp16.onnx
powershell -ExecutionPolicy Bypass -File src\setup_deps.ps1      # once
powershell -ExecutionPolicy Bypass -File src\build.ps1
```

### 4. Calibrate the lens (once per camera)

```powershell
powershell -ExecutionPolicy Bypass -File src\calibrate.ps1 --make-board board.png
# print at 100%, film it with the same camera (30-60 s, cover the corners), then:
powershell -ExecutionPolicy Bypass -File src\calibrate.ps1 --source C:\path\to\calib_videos --square 0.025 --show
```

This writes `fx, fy, cx, cy, dist` into `src/camera.yaml`. Aim for an RMS error below 0.5 px.

### 5. Run live

```powershell
powershell -ExecutionPolicy Bypass -File src\run.ps1
```

This plays every video in `videos/` in a live window. Space pauses, N skips to the next video, Q quits. It writes:

- `inference_output/<video>_inference.mp4`: the annotated video;
- `inference_output/<video>_distances.csv`: one row per detection (track id, class, confidence, box, ground-plane, size, fused and smoothed distance, uncertainty, horizon, pitch, camera height).

Options: `--source`, `--conf`, `--iou`, `--camera`, `--debug-lanes`, `--no-show`, `--cpu`. See [src/README.md](src/README.md).

---

## How distance estimation works

The camera is a single monocular camera, so distance is inferred from geometry rather than measured directly.

| Step | Method |
|---|---|
| Ground plane | The bottom-centre of the box touches a flat road. With the camera height `h` and pitch `θ`, the ray through that pixel meets the road at `Z = h / tan(θ + atan((v − c_y)/f_y))`. |
| Known size | `Z = f_y × H_real / h_px`, using per-class heights from `camera.yaml`. |
| Fusion | The two estimates are combined, each weighted by its expected error. Size alone is used when the bonnet hides the bottom of the object (shown with a `~`). |
| Dynamic horizon | Lane lines are found each frame (Canny + Hough); where they meet gives the horizon, so pitch changes from the car's movement are followed. |
| Auto camera height | For objects of a standard size (cones), the camera height that makes both estimates agree is learned as a running median. |
| Tracking | Detections are linked across frames by box overlap, and each object's distance is smoothed with a 1-D Kalman filter. |

**Accuracy depends on calibration.** Both estimates scale linearly with the focal length, so the default `hfov_deg: 110` is a placeholder until the lens is calibrated. Real object heights and the camera height should also be measured. Error grows with distance, and slopes break the flat-road assumption. Before relying on the numbers, validate them against cones placed at tape-measured distances.

---

## Current results

These were measured on the five development clips (1280×720) and need confirming on more data.

| Item | Result |
|---|---|
| Training data | 575 frames from 5 clips, trained with `--all` on **unreviewed** auto-labels |
| Validation (held-out `obstacle_course_trial2` clip) | best mAP50 **0.535**, mAP50-95 0.277, precision 0.576, recall 0.469 (epoch 16 of 46, early-stopped) |
| Stronger classes | cones, people, model houses |
| Weak classes | barriers (very few labels); chairs (low confidence) |
| C++ inference speed | about 16 ms per frame end to end, FP16 on CUDA (pre-processing 2 ms, model 14 ms, post-processing 0.3 ms) |
| Distance consistency | size and ground-plane estimates agree within about ±5% on cones (partly by design of the auto camera height); **absolute accuracy not yet validated** |
| Lens calibration | validated on synthetic data: fx recovered within 0.2%, k1 within 1%, 0.28 px RMS |

**Next steps:**
- review the labels, especially barriers and chairs, and retrain;
- calibrate the real camera and measure the props;
- record a ground-truth distance clip;
- add more varied footage (locations, lighting, times of day).

---

## Data protection

Some clips and frames show an identifiable person. Treat videos, extracted frames, inference outputs and calibration previews as **personal data under UK/EU GDPR**. Keep them internal, ensure there is consent or another lawful basis, and blur faces before sharing outside the team. The `.gitignore` excludes them from the repository; keep it that way.
#   o b j e c t - d e t e c t i o n - d i s t a n c e - e s t i m a t i o n -  
 