"""Benchmark the C++ inference app across devices and precisions; writes results.json + COMPARISON.md.

Configurations: CPU / CUDA / TensorRT  x  FP32 / FP16 / INT8.
  * Accuracy: mAP50 and mAP50-95 on the held-out 'obstacle_course_trial2' frames (never trained on),
    using Ultralytics' AP code, conf 0.001 / NMS IoU 0.7 like `yolo val`.
  * Speed: the 'house' clip (336 frames) through yolo_infer.exe, median / p95 after 10 warm-up frames.

Usage (from the project root, after src\\build.ps1 and benchmark\\quantize_int8.py):
    python benchmark/run_benchmark.py
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from ultralytics.utils.metrics import ap_per_class

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "benchmark"
EXE = ROOT / "src" / "build" / "yolo_infer.exe"
W = ROOT / "model_training" / "weights"
DATASET = ROOT / "annotation" / "dataset"
VAL_PREFIX = "obstacle_course_trial2_"
SPEED_VIDEO = ROOT / "videos" / "house_edited_4x.mp4"

CONFIGS = [  # id, device label, provider, precision flag, model file
    ("cpu_fp32", "CPU", "cpu", "fp32", "best.onnx"),
    ("cpu_fp16", "CPU", "cpu", "fp16", "best_fp16.onnx"),
    ("cpu_int8", "CPU", "cpu", "int8", "best_int8.onnx"),
    ("cuda_fp32", "GPU (CUDA)", "cuda", "fp32", "best.onnx"),
    ("cuda_fp16", "GPU (CUDA)", "cuda", "fp16", "best_fp16.onnx"),
    ("cuda_int8", "GPU (CUDA)", "cuda", "int8", "best_int8.onnx"),
    ("trt_fp32", "GPU (TensorRT)", "tensorrt", "fp32", "best.onnx"),
    ("trt_fp16", "GPU (TensorRT)", "tensorrt", "fp16", "best.onnx"),
    ("trt_int8", "GPU (TensorRT)", "tensorrt", "int8", "best_int8.onnx"),
]


def run(args):
    env = dict(os.environ, PATH=r"C:\msys64\ucrt64\bin;" + os.environ["PATH"],
               TMP=str(BENCH / "tmp"), TEMP=str(BENCH / "tmp"))
    t = time.perf_counter()
    p = subprocess.run([str(EXE)] + args, cwd=ROOT, env=env, capture_output=True, text=True)
    out = p.stdout + p.stderr
    if p.returncode:
        raise RuntimeError(f"yolo_infer failed ({p.returncode}):\n{out[-2000:]}")
    return out, time.perf_counter() - t


def parse_common(out):
    prov = re.search(r"provider (.+?)  session ready in ([\d.]+) s", out)
    return {"provider_used": prov.group(1), "session_s": float(prov.group(2))}


def parse_timing(out):
    m = re.search(r"median total ([\d.]+) ms, p95 ([\d.]+) ms, median model ([\d.]+) ms", out)
    return {"median_ms": float(m.group(1)), "p95_ms": float(m.group(2)), "model_ms": float(m.group(3))}


def load_txt(path, with_conf):
    if not path.exists() or path.stat().st_size == 0:
        return np.zeros((0, 6 if with_conf else 5))
    a = np.loadtxt(path, ndmin=2)
    return a


def xywh2xyxy(b):
    return np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)


def box_iou(a, b):
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def match(pred_cls, gt_cls, iou, thresholds):
    """Greedy matching, same as Ultralytics DetectionValidator.match_predictions."""
    correct = np.zeros((len(pred_cls), len(thresholds)), bool)
    iou = iou * (gt_cls[:, None] == pred_cls[None, :])
    for i, t in enumerate(thresholds):
        m = np.array(np.nonzero(iou >= t)).T
        if m.shape[0]:
            if m.shape[0] > 1:
                m = m[iou[m[:, 0], m[:, 1]].argsort()[::-1]]
                m = m[np.unique(m[:, 1], return_index=True)[1]]
                m = m[np.unique(m[:, 0], return_index=True)[1]]
            correct[m[:, 1], i] = True
    return correct


def evaluate(pred_dir, stems, names):
    thr = np.linspace(0.5, 0.95, 10)
    tps, confs, pcls, tcls = [], [], [], []
    for s in stems:
        gt = load_txt(DATASET / "labels" / f"{s}.txt", False)
        pr = load_txt(pred_dir / f"{s}.txt", True)
        tcls.append(gt[:, 0])
        if len(pr) == 0:
            continue
        if len(gt):
            # 1280x720 frames: scale normalised boxes to pixels so IoU is aspect-correct
            sc = np.array([1280, 720, 1280, 720])
            iou = box_iou(xywh2xyxy(gt[:, 1:5] * sc), xywh2xyxy(pr[:, 1:5] * sc))
            tps.append(match(pr[:, 0], gt[:, 0], iou, thr))
        else:
            tps.append(np.zeros((len(pr), len(thr)), bool))
        confs.append(pr[:, 5])
        pcls.append(pr[:, 0])
    tcls = np.concatenate(tcls)
    if not tps:
        return {"map50": 0.0, "map": 0.0, "per_class_ap50": {}}
    res = ap_per_class(np.concatenate(tps), np.concatenate(confs), np.concatenate(pcls), tcls)
    p, r, ap, classes = res[2], res[3], res[5], res[6]
    return {"map50": float(ap[:, 0].mean()), "map": float(ap.mean()), "precision": float(p.mean()),
            "recall": float(r.mean()),
            "per_class_ap50": {names[int(c)]: float(ap[i, 0]) for i, c in enumerate(classes)},
            "gt_counts": {names[int(c)]: int((tcls == c).sum()) for c in np.unique(tcls)}}


def main():
    if not EXE.exists():
        sys.exit("Build first: src\\build.ps1")
    names = [l.strip() for l in (DATASET / "classes.txt").read_text().splitlines() if l.strip()]
    (BENCH / "tmp").mkdir(parents=True, exist_ok=True)
    val_dir = BENCH / "val_images"
    if not val_dir.exists():
        val_dir.mkdir()
        for p in sorted((DATASET / "images").glob(VAL_PREFIX + "*.jpg")):
            shutil.copy2(p, val_dir / p.name)
    stems = sorted(p.stem for p in val_dir.glob("*.jpg"))
    print(f"Validation: {len(stems)} held-out frames")

    results = []
    for cid, device, prov, prec, model in CONFIGS:
        mpath = W / model
        if not mpath.exists():
            print(f"skip {cid}: {mpath.name} missing")
            continue
        print(f"== {cid}")
        pred_dir = BENCH / "preds" / cid
        shutil.rmtree(pred_dir, ignore_errors=True)
        common = ["--provider", prov, "--precision", prec, "--model", str(mpath.relative_to(ROOT))]
        # 1) accuracy run (first TensorRT run also builds + caches the engine)
        out, _ = run(common + ["--images", str(val_dir), "--pred-dir", str(pred_dir), "--conf", "0.001",
                               "--iou", "0.7"])
        first = parse_common(out)
        acc = evaluate(pred_dir, stems, names)
        # 2) speed run on a video, normal thresholds, engine now cached
        vid_out = BENCH / "runs" / cid
        out, wall = run(common + ["--source", str(SPEED_VIDEO), "--no-show", "--out", str(vid_out)])
        second = parse_common(out)
        timing = parse_timing(out)
        r = {"id": cid, "device": device, "provider_flag": prov, "precision": prec.upper(), "model": model,
             "model_mb": round(mpath.stat().st_size / 2**20, 1), **acc, **timing,
             "first_session_s": first["session_s"], "session_s": second["session_s"],
             "provider_used": second["provider_used"]}
        results.append(r)
        print(f"   {r['provider_used']}: mAP50 {r['map50']:.3f}  mAP50-95 {r['map']:.3f}  "
              f"median {r['median_ms']:.1f} ms  model {r['model_ms']:.1f} ms  start {r['session_s']:.1f} s")
        shutil.rmtree(vid_out, ignore_errors=True)

    meta = {"date": time.strftime("%d/%m/%Y %H:%M"), "val_frames": len(stems), "speed_video": SPEED_VIDEO.name}
    (BENCH / "results.json").write_text(json.dumps({"meta": meta, "results": results}, indent=1))
    print(f"Wrote {BENCH / 'results.json'}")


if __name__ == "__main__":
    main()
