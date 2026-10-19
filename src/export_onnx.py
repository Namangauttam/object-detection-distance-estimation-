"""Export the trained YOLOv8 model to ONNX for the C++ inference app.

Writes, next to the weights:
    best.onnx       FP32 (works on CPU and GPU)
    best_fp16.onnx  FP16 (GPU only, ~30% faster; used by yolo_infer by default when present)

Usage (from the project root, with the Python that has ultralytics installed):
    python src/export_onnx.py
    python src/export_onnx.py --weights model_training/runs/yolov8s_ft/weights/best.pt --no-fp16
"""
import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "model_training" / "weights" / "best.pt"))
    ap.add_argument("--imgsz", type=int, default=1280, help="must match the training image size")
    ap.add_argument("--no-fp16", action="store_true", help="skip the FP16 export")
    args = ap.parse_args()

    weights = Path(args.weights)
    # Static 1x3xNxN input; class names are stored in the ONNX metadata and read by the C++ app.
    common = dict(format="onnx", imgsz=args.imgsz, opset=17, simplify=True, dynamic=False)

    if not args.no_fp16 and torch.cuda.is_available():
        out = Path(YOLO(weights).export(**common, quantize=16, device=0))
        fp16 = weights.with_name(weights.stem + "_fp16.onnx")
        shutil.move(out, fp16)
        print(f"Exported FP16: {fp16}")

    out = YOLO(weights).export(**common, device="cpu")
    print(f"Exported FP32: {out}")


if __name__ == "__main__":
    main()
