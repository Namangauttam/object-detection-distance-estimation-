"""Write train/val image lists for YOLO training, splitting by video (no leakage).

Usage:
    python split_dataset.py                                  # val = obstacle_course_trial2
    python split_dataset.py --val pedestrian obstacle_course_trial2
    python split_dataset.py --reviewed-only                  # only images marked reviewed
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", nargs="+", default=["obstacle_course_trial2"],
                    help="video name prefixes held out for validation")
    ap.add_argument("--reviewed-only", action="store_true")
    args = ap.parse_args()

    review = json.loads((DATASET / "review.json").read_text()) if (DATASET / "review.json").exists() else {}
    train, val = [], []
    for p in sorted((DATASET / "images").glob("*.jpg")):
        if args.reviewed_only and not review.get(p.name):
            continue
        video = p.stem.rsplit("_", 1)[0]
        (val if video in args.val else train).append(p.resolve().as_posix())

    (DATASET / "train.txt").write_text("\n".join(train) + "\n")
    (DATASET / "val.txt").write_text("\n".join(val) + "\n")

    names = [l.strip() for l in (DATASET / "classes.txt").read_text().splitlines() if l.strip()]
    lines = [f"path: {DATASET.as_posix()}", "train: train.txt", "val: val.txt", "names:"] + \
            [f"  {i}: {n}" for i, n in enumerate(names)]
    (DATASET / "data.yaml").write_text("\n".join(lines) + "\n")
    print(f"train: {len(train)}  val: {len(val)}  classes: {names}")


if __name__ == "__main__":
    main()
