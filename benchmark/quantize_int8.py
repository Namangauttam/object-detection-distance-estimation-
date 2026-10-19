"""Static INT8 (QDQ) quantisation of the exported YOLOv8 ONNX model.

Calibrates activation ranges on frames from the TRAINING videos (never the held-out validation clip),
pre-processed exactly like the C++ app (letterbox to 1280, RGB, /255). The box-decode part of the
detection head (DFL softmax, anchor maths, sigmoid, concat) stays in float: quantising coordinates up
to 1280 px to 8 bits would cost accuracy for no real speed gain.

The QDQ model runs on ONNX Runtime CPU (int8 kernels) and on TensorRT (--precision int8).

Usage (from the project root):
    python benchmark/quantize_int8.py
"""
import argparse
import os
import random
import tempfile
from pathlib import Path

import cv2
import numpy as np
import onnx
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                      quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "model_training" / "weights"
IMAGES = ROOT / "annotation" / "dataset" / "images"
HELD_OUT = "obstacle_course_trial2"


def letterbox(img, size):
    h, w = img.shape[:2]
    r = min(size / w, size / h)
    nw, nh = round(w * r), round(h * r)
    out = np.full((size, size, 3), 114, np.uint8)
    px, py = round((size - nw) / 2 - 0.1), round((size - nh) / 2 - 0.1)
    out[py:py + nh, px:px + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    x = cv2.cvtColor(out, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return x.transpose(2, 0, 1)[None]


class FrameReader(CalibrationDataReader):
    def __init__(self, files, input_name, size):
        self.files, self.input_name, self.size, self.i = files, input_name, size, 0

    def get_next(self):
        if self.i >= len(self.files):
            return None
        x = letterbox(cv2.imread(str(self.files[self.i])), self.size)
        self.i += 1
        return {self.input_name: x}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(WEIGHTS / "best.onnx"))
    ap.add_argument("--out", default=str(WEIGHTS / "best_int8.onnx"))
    ap.add_argument("--calib-images", type=int, default=60)  # not a multiple of 8: ORT bug
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model = onnx.load(args.model)
    inp = model.graph.input[0]
    size = inp.type.tensor_type.shape.dim[2].dim_value or 1280

    # Keep the head's decode ops in float; its convolutions are still quantised.
    exclude = [n.name for n in model.graph.node
               if n.name.startswith("/model.22/") and (n.op_type != "Conv" or "/dfl/" in n.name)]
    print(f"Excluding {len(exclude)} head decode nodes from quantisation")

    files = sorted(p for p in IMAGES.glob("*.jpg") if not p.name.startswith(HELD_OUT))
    random.Random(args.seed).shuffle(files)
    files = files[:args.calib_images]
    print(f"Calibrating on {len(files)} training frames (held-out '{HELD_OUT}' excluded)")

    with tempfile.TemporaryDirectory(dir=Path(args.out).parent) as tmp:
        pre = os.path.join(tmp, "pre.onnx")
        quant_pre_process(args.model, pre, skip_symbolic_shape=True)
        quantize_static(
            pre, args.out, FrameReader(files, inp.name, size),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,   # symmetric int8 activations: what TensorRT expects
            weight_type=QuantType.QInt8,
            per_channel=True,
            calibrate_method=CalibrationMethod.MinMax,
            nodes_to_exclude=exclude,
            extra_options={"ActivationSymmetric": True, "WeightSymmetric": True,
                           "CalibMaxIntermediateOutputs": 8, "CalibTensorRangeSymmetric": True},
        )
    # Keep the class names for the C++ app.
    q = onnx.load(args.out)
    for p in model.metadata_props:
        e = q.metadata_props.add()
        e.key, e.value = p.key, p.value
    onnx.save(q, args.out)
    print(f"Wrote {args.out} ({Path(args.out).stat().st_size / 2**20:.1f} MB)")


if __name__ == "__main__":
    main()
