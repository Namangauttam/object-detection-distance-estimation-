"""Run the fine-tuned model on videos and save annotated copies.

Usage:
    python predict_video.py                                   # all videos in ../videos
    python predict_video.py --source ../videos/cones_edited_4x.mp4 --conf 0.3
"""
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(HERE / "weights" / "best.pt"))
    ap.add_argument("--source", default=str(HERE.parent / "videos"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--out", default=str(HERE / "predictions"))
    args = ap.parse_args()

    model = YOLO(args.weights)
    src = Path(args.source)
    videos = sorted(src.glob("*.mp4")) if src.is_dir() else [src]
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    for video in videos:
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = out_dir / f"{video.stem}_pred.mp4"
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        counts = {}
        for r in model.predict(str(video), conf=args.conf, imgsz=args.imgsz, stream=True, verbose=False):
            writer.write(r.plot(line_width=2))
            for c in r.boxes.cls.tolist():
                counts[r.names[int(c)]] = counts.get(r.names[int(c)], 0) + 1
        writer.release()
        cap.release()
        print(f"{video.name} -> {out_path.name}   detections: {counts}")


if __name__ == "__main__":
    main()
