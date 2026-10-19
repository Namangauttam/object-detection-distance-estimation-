"""Build docs/demo.gif (shown at the top of the README) from the C++ inference output videos.

Only clips WITHOUT identifiable people are used (the repo may be shared; footage of people is personal data).

Usage (from the project root, after src\\run.ps1 has produced inference_output/):
    python docs/make_demo_gif.py
"""
from pathlib import Path

import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "demo.gif"
SEGMENTS = [  # (video, start s, end s)
    ("cones_edited_4x_inference.mp4", 2.5, 10.5),
    ("obstacle_course_trial2_4x_inference.mp4", 0.5, 9.5),
]
WIDTH, FPS = 560, 6
CROP_BOTTOM = 0.78  # keep the top 78% of the frame


def main():
    frames = []
    for name, t0, t1 in SEGMENTS:
        cap = cv2.VideoCapture(str(ROOT / "inference_output" / name))
        if not cap.isOpened():
            raise SystemExit(f"Missing inference_output/{name} - run src\\run.ps1 first")
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        step = src_fps / FPS
        i, next_keep = 0, t0 * src_fps
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(next_keep))
        i = int(next_keep)
        while i <= t1 * src_fps:
            ok, f = cap.read()
            if not ok:
                break
            if i >= next_keep:
                f = f[: int(f.shape[0] * CROP_BOTTOM)]  # drop the car bonnet: fewer bytes, more road
                h = round(f.shape[0] * WIDTH / f.shape[1])
                rgb = cv2.cvtColor(cv2.resize(f, (WIDTH, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb).quantize(colors=128, method=Image.Quantize.FASTOCTREE,
                                                            dither=Image.Dither.NONE))
                next_keep += step
            i += 1
    frames[0].save(OUT, save_all=True, append_images=frames[1:], duration=int(1000 / FPS), loop=0,
                   optimize=True, disposal=1)
    print(f"Wrote {OUT} ({len(frames)} frames, {OUT.stat().st_size / 2**20:.1f} MB)")


if __name__ == "__main__":
    main()
