"""Extract frames from the videos into dataset/images.

Usage:
    python extract_frames.py              # every 4th frame (videos are 4x speed)
    python extract_frames.py --every 2    # denser sampling
"""
import argparse
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
VIDEOS = ROOT.parent / "videos"
IMAGES = ROOT / "dataset" / "images"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=4, help="keep one frame in N")
    ap.add_argument("--videos", type=Path, default=VIDEOS)
    args = ap.parse_args()

    IMAGES.mkdir(parents=True, exist_ok=True)
    total = 0
    for video in sorted(args.videos.glob("*.mp4")):
        cap = cv2.VideoCapture(str(video))
        stem = video.stem.replace("_edited", "").replace("_4x", "")
        i = saved = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % args.every == 0:
                out = IMAGES / f"{stem}_{i:05d}.jpg"
                if not out.exists():
                    cv2.imwrite(str(out), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved += 1
            i += 1
        cap.release()
        print(f"{video.name}: {i} frames read, {saved} kept")
        total += saved
    print(f"Total: {total} images in {IMAGES}")


if __name__ == "__main__":
    main()
