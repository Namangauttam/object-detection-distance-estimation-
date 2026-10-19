"""Pre-annotate dataset/images with YOLO-World and write YOLO-format labels.

Existing label files are NOT overwritten (so your manual fixes are safe)
unless you pass --overwrite.

Usage:
    python auto_label.py
    python auto_label.py --overwrite --conf-scale 0.8
"""
import argparse
from pathlib import Path

import torch
import torchvision
from ultralytics import YOLOWorld

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset"
IMAGES = DATASET / "images"
LABELS = DATASET / "labels"
CLASSES_FILE = DATASET / "classes.txt"
MODEL = ROOT / "models" / "yolov8x-worldv2.pt"

BASE_CLASSES = ["cone", "person", "chair", "house", "barrier"]

# Several text prompts per class; detections are merged back to the class id.
PROMPTS = {
    "traffic cone": "cone",
    "orange cone": "cone",
    "person": "person",
    "pedestrian": "person",
    "chair": "chair",
    "toy house": "house",
    "small white model house": "house",
    "red road barrier": "barrier",
    "plastic barricade": "barrier",
}

# Per-class minimum confidence (tuned low: deleting a false box is quicker than drawing one).
CONF = {"cone": 0.20, "person": 0.30, "chair": 0.25, "house": 0.15, "barrier": 0.15}

# The car bonnet occupies the bottom of the frame and reflects objects ahead.
# Drop boxes whose top edge starts below this fraction of image height.
BONNET_TOP = 0.74


def load_classes():
    if CLASSES_FILE.exists():
        names = [l.strip() for l in CLASSES_FILE.read_text().splitlines() if l.strip()]
    else:
        names = list(BASE_CLASSES)
        CLASSES_FILE.write_text("\n".join(names) + "\n")
    return names


def write_data_yaml(names):
    lines = [
        f"path: {DATASET.as_posix()}",
        "train: images  # TODO: split into train/val before training",
        "val: images",
        "names:",
    ] + [f"  {i}: {n}" for i, n in enumerate(names)]
    (DATASET / "data.yaml").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true", help="replace existing label files")
    ap.add_argument("--conf-scale", type=float, default=1.0, help="multiply all thresholds")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    names = load_classes()
    write_data_yaml(names)
    LABELS.mkdir(parents=True, exist_ok=True)

    prompts = list(PROMPTS)
    prompt_to_cls = [names.index(PROMPTS[p]) for p in prompts]
    conf_by_cls = {names.index(k): v * args.conf_scale for k, v in CONF.items()}
    min_conf = min(conf_by_cls.values())

    todo = [p for p in sorted(IMAGES.glob("*.jpg"))
            if args.overwrite or not (LABELS / f"{p.stem}.txt").exists()]
    print(f"{len(todo)} images to label")
    if not todo:
        return

    model = YOLOWorld(str(MODEL))
    model.set_classes(prompts)

    counts = [0] * len(names)
    for start in range(0, len(todo), args.batch):
        chunk = todo[start:start + args.batch]
        results = model.predict([str(p) for p in chunk], conf=min_conf, imgsz=args.imgsz,
                                verbose=False)
        for path, r in zip(chunk, results):
            h, w = r.orig_shape
            lines = []
            if len(r.boxes):
                xyxy = r.boxes.xyxy.cpu()
                conf = r.boxes.conf.cpu()
                cls = torch.tensor([prompt_to_cls[int(c)] for c in r.boxes.cls.cpu()])
                keep = torch.tensor([float(cf) >= conf_by_cls.get(int(c), 0.25)
                                     and float(b[1]) / h < BONNET_TOP
                                     for b, cf, c in zip(xyxy, conf, cls)], dtype=torch.bool)
                xyxy, conf, cls = xyxy[keep], conf[keep], cls[keep]
                idx = torchvision.ops.batched_nms(xyxy, conf, cls, 0.5)
                for i in idx.tolist():
                    x1, y1, x2, y2 = xyxy[i].tolist()
                    c = int(cls[i])
                    counts[c] += 1
                    lines.append(f"{c} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                                 f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
            (LABELS / f"{path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        print(f"  {min(start + args.batch, len(todo))}/{len(todo)}")

    print("Boxes per class:", {n: counts[i] for i, n in enumerate(names)})


if __name__ == "__main__":
    main()
