"""Fine-tune YOLOv8s on the reviewed annotations in ../annotation/dataset.

Usage:
    python train.py                       # reviewed images only, default settings
    python train.py --all                 # include unreviewed (auto-labelled) images too
    python train.py --epochs 150 --batch 4
    python train.py --resume              # continue an interrupted run

Outputs:
    runs/<name>/            Ultralytics run (curves, confusion matrix, val images)
    weights/best.pt         copy of the best checkpoint
"""
import argparse
import json
import random
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
DATASET = HERE.parent / "annotation" / "dataset"
SPLITS = HERE / "splits"


def build_split(val_videos, use_all, val_frac, seed):
    names = [l.strip() for l in (DATASET / "classes.txt").read_text().splitlines() if l.strip()]
    review_file = DATASET / "review.json"
    review = json.loads(review_file.read_text()) if review_file.exists() else {}

    images = sorted((DATASET / "images").glob("*.jpg"))
    if not use_all:
        images = [p for p in images if review.get(p.name)]
        if not images:
            raise SystemExit("No reviewed images yet. Review some in the annotation tool, "
                             "or run with --all to train on the raw auto-labels.")

    by_video = {}
    for p in images:
        by_video.setdefault(p.stem.rsplit("_", 1)[0], []).append(p)

    train, val = [], []
    held_out = [v for v in val_videos if v in by_video]
    if held_out:
        # Hold out whole videos so near-identical neighbouring frames can't leak into val.
        for v, ps in by_video.items():
            (val if v in held_out else train).extend(ps)
    else:
        # Fallback: last chunk of each video (contiguous, to limit leakage).
        print(f"Held-out video(s) {val_videos} not available; using last {val_frac:.0%} of each video.")
        for ps in by_video.values():
            k = max(1, round(len(ps) * val_frac)) if len(ps) > 3 else 0
            train.extend(ps[:len(ps) - k])
            val.extend(ps[len(ps) - k:])

    if not val:
        random.Random(seed).shuffle(train)
        k = max(1, len(train) // 7)
        val, train = train[:k], train[k:]

    SPLITS.mkdir(exist_ok=True)
    (SPLITS / "train.txt").write_text("\n".join(p.as_posix() for p in train) + "\n")
    (SPLITS / "val.txt").write_text("\n".join(p.as_posix() for p in val) + "\n")
    yaml = [f"path: {SPLITS.as_posix()}", "train: train.txt", "val: val.txt", "names:"] + \
           [f"  {i}: {n}" for i, n in enumerate(names)]
    (SPLITS / "data.yaml").write_text("\n".join(yaml) + "\n")

    print(f"Classes: {names}")
    print(f"Train: {len(train)} images  |  Val: {len(val)} images "
          f"({'held-out: ' + ', '.join(held_out) if held_out else 'per-video tail'})")
    return SPLITS / "data.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s.pt", help="starting weights")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=1280, help="high res helps small, distant objects")
    ap.add_argument("--batch", type=int, default=8, help="lower to 4 if you get CUDA out-of-memory")
    ap.add_argument("--patience", type=int, default=30, help="early-stop after N epochs without improvement")
    # On Windows each dataloader worker is a new process that reloads torch/CUDA DLLs
    # (several GB of virtual memory each) -> "WinError 1455 paging file too small".
    ap.add_argument("--workers", type=int, default=0 if sys.platform == "win32" else 4,
                    help="dataloader processes; on Windows try 2 only after enlarging the page file")
    ap.add_argument("--cache", default="disk", choices=["disk", "ram", "none"],
                    help="image cache; 'ram' is fastest but uses ~1.5 GB extra memory")
    ap.add_argument("--name", default="yolov8s_ft")
    ap.add_argument("--val-videos", nargs="+", default=["obstacle_course_trial2"])
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--all", action="store_true", help="also use images not yet marked reviewed")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.resume:
        last = HERE / "runs" / args.name / "weights" / "last.pt"
        results_dir = Path(YOLO(str(last)).train(resume=True).save_dir)
    else:
        data = build_split(args.val_videos, args.all, args.val_frac, args.seed)
        model = YOLO(args.model)
        model.train(
            data=str(data),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            patience=args.patience,
            workers=args.workers,
            project=str(HERE / "runs"),
            name=args.name,
            exist_ok=True,
            seed=args.seed,
            device=0,
            cache=False if args.cache == "none" else args.cache,
            # Augmentation: road scenes are left/right symmetric but never upside down.
            fliplr=0.5, flipud=0.0, degrees=0.0,
            hsv_h=0.015, hsv_s=0.6, hsv_v=0.5,   # strong lighting changes: sun, shade, glare
            scale=0.5, translate=0.1,
            mosaic=1.0, close_mosaic=10,
            plots=True,
        )
        results_dir = HERE / "runs" / args.name

    best = results_dir / "weights" / "best.pt"
    if best.exists():
        out = HERE / "weights"
        out.mkdir(exist_ok=True)
        shutil.copy2(best, out / "best.pt")
        print(f"\nBest weights copied to {out / 'best.pt'}")
        metrics = YOLO(str(best)).val(data=str(SPLITS / "data.yaml"), imgsz=args.imgsz,
                                      batch=args.batch, plots=False, verbose=True)
        print(f"Final val mAP50: {metrics.box.map50:.3f}  mAP50-95: {metrics.box.map:.3f}")


if __name__ == "__main__":
    main()
