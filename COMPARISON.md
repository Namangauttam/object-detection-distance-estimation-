# Inference benchmark: CPU vs GPU, FP32 vs FP16 vs INT8

**Date:** 20/10/2026 · **Model:** YOLOv8s fine-tuned (6 classes), 1280×1280 input · **App:** C++ `yolo_infer.exe` (ONNX Runtime 1.31)

## Summary

| | Recommendation |
|---|---|
| **Use for live runs** | **GPU, TensorRT, FP16.** Fastest (2.3× faster than TensorRT FP32 and 1.9× faster than CUDA FP16 in this run) with **no accuracy loss** (mAP50 0.529, the same as FP32). |
| No TensorRT available | GPU, CUDA, FP16. Same accuracy, about 1.5× faster than CUDA FP32. |
| CPU only | CPU FP16 or FP32 (equal). About 2.7 FPS, which is too slow for live use but fine for offline checks. |
| **INT8 (as quantised here)** | **Not recommended.** It is 4× smaller, but **slower** on every device and loses 2.2–2.9 mAP50 points, mostly on chairs. See [why INT8 is slower](#why-int8-is-slower-and-less-accurate-here). |

FP16 costs nothing in accuracy on this model, so FP16 is the right default on the GPU.

---

## Results

Accuracy is measured on 164 held-out frames from `obstacle_course_trial2`, which was never used in training, with conf 0.001 and NMS IoU 0.7 (the same as `yolo val`). Speed is measured on the `house` clip (336 frames): the median and 95th percentile per frame after 10 warm-up frames, covering pre-processing, model and post-processing. FPS = 1000 / median.

| Device | Precision | Model size | mAP50 | mAP50-95 | Δ mAP50 vs FP32 | Median ms/frame | p95 ms | Model-only ms | FPS | Start-up (cached) |
|---|---|---|---|---|---|---|---|---|---|---|
| CPU | FP32 | 43.2 MB | 0.529 | 0.274 | — | 368.6 | 399.9 | 364.1 | 2.7 | 0.2 s |
| CPU | FP16 | 21.6 MB | 0.529 | 0.277 | 0.000 | 351.4 | 363.3 | 348.1 | 2.8 | 0.2 s |
| CPU | INT8 | 11.6 MB | 0.507 | 0.261 | −0.022 | 806.5 | 880.4 | 802.6 | 1.2 | 0.2 s |
| GPU, CUDA | FP32 | 43.2 MB | 0.529 | 0.274 | — | 55.1 | 59.1 | 51.1 | 18.2 | 0.5 s |
| GPU, CUDA | FP16 | 21.6 MB | 0.529 | 0.273 | 0.000 | 36.0 | 39.0 | 32.8 | 27.8 | 0.4 s |
| GPU, CUDA | INT8 | 11.6 MB | 0.506 | 0.258 | −0.023 | 92.5 | 103.1 | 88.9 | 10.8 | 0.6 s |
| GPU, TensorRT | FP32 | 43.2 MB¹ | 0.529 | 0.274 | 0.000 | 45.1 | 50.0 | 41.5 | 22.2 | 2.5 s |
| **GPU, TensorRT** | **FP16** | 43.2 MB¹ | **0.529** | 0.271 | **0.000** | **19.4** | **22.6** | **15.7** | **51.5** | 1.1 s |
| GPU, TensorRT | INT8 | 11.6 MB | 0.500 | 0.251 | −0.029 | 30.9 | 34.7 | 27.3 | 32.3 | 8.7 s |

¹ TensorRT builds its own engine from the FP32 ONNX file. The cached engine is 22.6 MB for FP16.

**First-run engine build (TensorRT only, cached afterwards):** FP32 109 s, INT8 226 s. FP16 had already been built in an earlier session; that build took about 50 s.

### ⚠️ Speed numbers were affected by other load on the laptop

During this run, a large Hugging Face model download was using the machine, and another GPU job (`det.py`) started just after it. Every timing came out about 2× slower than isolated measurements taken an hour earlier on an idle machine:

| Isolated, idle machine (same clip, back to back) | Model-only ms | Total ms/frame | FPS |
|---|---|---|---|
| GPU, CUDA, FP16 | 15.8 | 17.7 | about 56 |
| GPU, TensorRT, FP16 | 7.2 | 9.7 | about 100 |

The **ranking and ratios are consistent** between the two measurements: TensorRT FP16 is about 2× CUDA FP16. Treat the absolute numbers in the main table as pessimistic. To refresh them, re-run on an idle machine (see [Reproduce](#reproduce)). Accuracy figures are not affected by load.

---

## Accuracy per class (AP50)

| Class | Ground-truth boxes | FP32 (any device) | FP16 (any device) | INT8 (CPU / CUDA / TensorRT) |
|---|---|---|---|---|
| cone | 83 | 0.853 | 0.853 | 0.844 / 0.857 / 0.850 |
| chair | 9 | 0.814 | 0.814 | **0.678 / 0.684 / 0.662** |
| house | 21 | 0.646 | 0.646 | 0.645 / 0.642 / 0.647 |
| barrier | 105 | 0.333 | 0.333–0.335 | 0.367 / 0.349 / 0.343 |
| TRAFFIC | 25 | 0.000 | 0.000 | 0.000 |
| **mAP50 excluding TRAFFIC** | | **0.661** | **0.662** | 0.634 / 0.633 / 0.625 |

- **FP16 matches FP32** to within ±0.002 on every class.
- **INT8 mainly hurts chairs** (−0.13 to −0.15 AP50). Chairs are small, low-contrast objects and there are only 9 in the validation set, so this number is noisy.
- **TRAFFIC scores 0 at every precision.** It's the class added by hand in the annotation tool, and it has too few training labels for the model to learn it. That pulls overall mAP down, so mAP50 excluding TRAFFIC is shown as well.
- **No `person` appears in the validation clip,** so person accuracy isn't measured here.

---

## Why INT8 is slower and less accurate here

The INT8 model uses **post-training static quantisation**: ONNX Runtime QDQ format, symmetric int8, per-channel weights, MinMax calibration on 60 training frames. The detection head's box-decode layers are kept in float.

1. **SiLU activations break the INT8 chain.** YOLOv8 uses SiLU (`x · sigmoid(x)`) after almost every convolution. ONNX Runtime's CPU and CUDA providers can't run it in int8, so each layer converts int8 → float → SiLU → int8. That adds hundreds of extra conversion operations, which outweigh the cheaper int8 convolutions.
2. **The CUDA provider has no int8 convolution kernels,** so QDQ models run in float with extra conversion steps. That makes it the slowest GPU option.
3. **TensorRT fuses better but still adds reformatting layers** around the float SiLU and head layers, so INT8 (27 ms) lands between FP32 (42 ms) and FP16 (16 ms).
4. **Accuracy:** MinMax calibration with 60 frames is crude, and 8-bit activations lose detail on small objects such as chairs.

**If INT8 is needed** (for example on a Jetson or another edge device), better options are:
- quantisation-aware training, or NVIDIA ModelOpt / pytorch-quantization with SiLU-aware QDQ placement;
- entropy or percentile calibration on more frames;
- keeping the first and last layers in FP16.

On this laptop GPU, FP16 is the better trade-off.

---

## Test setup

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5050 Laptop (8 GB, Blackwell sm_120), mains power, Windows *Balanced* power plan |
| CPU | Intel, 24 logical cores (ONNX Runtime default thread count) |
| Software | ONNX Runtime 1.31 GPU (CUDA 12), TensorRT 10.16.1, CUDA 12.8 / cuDNN 9 runtime (from PyTorch 2.11), OpenCV 5.0, g++ 16.2 |
| Models | `best.onnx` (FP32), `best_fp16.onnx` (FP16 export), `best_int8.onnx` (QDQ, from [quantize_int8.py](quantize_int8.py)) |
| TensorRT precision | FP32 = TF32 allowed by default; FP16 = `trt_fp16_enable`; INT8 = `trt_int8_enable` with explicit QDQ, FP16 for the remaining layers |
| Timing | `std::chrono` around the model's `detect()` call (pre-processing, inference, post-processing); excludes video decoding, drawing and video writing |
| Evaluator check | CUDA FP32 gives mAP50 0.529, against 0.535 reported by Ultralytics during training (small NMS and letterbox differences), which confirms the evaluator |

### Accuracy caveats

- The labels are **unreviewed YOLO-World auto-labels**, so absolute mAP is approximate. The comparison *between* precisions is valid, because every configuration is scored against the same labels.
- The validation set is a single 164-frame clip. Classes with few boxes (chair 9, house 21) have noisy AP.

---

## Reproduce

From the project root, with the GPU idle:

```powershell
python benchmark\quantize_int8.py     # writes model_training\weights\best_int8.onnx
python benchmark\run_benchmark.py     # all 9 configurations -> benchmark\results.json
```

Raw numbers are in [results.json](results.json). Any single configuration can be run by hand:

```powershell
powershell -ExecutionPolicy Bypass -File src\run.ps1 --provider tensorrt --precision fp16 --no-show
powershell -ExecutionPolicy Bypass -File src\run.ps1 --provider cuda --model model_training\weights\best_fp16.onnx --no-show
powershell -ExecutionPolicy Bypass -File src\run.ps1 --provider cpu --model model_training\weights\best_int8.onnx --no-show
```
