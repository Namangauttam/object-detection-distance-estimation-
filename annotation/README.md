# Annotation pipeline

Front-camera frames → YOLO-World pre-labels → manual review in a local web tool → YOLO dataset.

## Folder layout

```
annotation/
  extract_frames.py    # videos/*.mp4 -> dataset/images/*.jpg
  auto_label.py        # YOLO-World pre-labels -> dataset/labels/*.txt (YOLO format)
  server.py            # local review tool at http://localhost:8765
  split_dataset.py     # train/val lists + data.yaml for training
  tool/index.html      # the review UI
  models/              # yolov8x-worldv2.pt
  dataset/
    images/            # <video>_<frame>.jpg
    labels/            # <video>_<frame>.txt   "class cx cy w h" (normalised 0-1)
    classes.txt        # one class per line; line number = class id
    review.json        # which images you have marked reviewed
    data.yaml          # Ultralytics dataset config
```

Classes: `0 cone`, `1 person`, `2 chair`, `3 house`, `4 barrier` (you can add more in the tool).

## Steps

```bash
cd annotation
python extract_frames.py          # every 4th frame (use --every 2 for more)
python auto_label.py              # skips images that already have labels; --overwrite to redo
python server.py                  # open http://localhost:8765
```

Running `auto_label.py` again never overwrites your corrected labels unless you pass `--overwrite`.

## Reviewing in the tool

1. Filter **Unreviewed**, then work through the images.
2. **Wrong box:** click it, then press `Del` (or `X`).
3. **Wrong class:** click the box, then press `1`–`9`, or use the dropdown in the right panel.
4. **Missed object:** pick a class (`1`–`9`), then drag a box. Scroll to zoom in on small, distant objects.
5. **New category:** type a name under *Classes*, then click **Add**. The new class gets the next id.
6. Press `Enter` to mark the image reviewed and move to the next one.

Edits save automatically. `P` copies boxes from the previous frame of the same video, which helps with objects the model missed in consecutive frames. Afterwards, adjust the copied boxes slightly.

Rules of thumb:
- Box every visible instance, including small distant ones you can identify confidently.
- Do **not** box reflections of objects on the car bonnet.
- If an image genuinely has no objects, leave it empty and still mark it reviewed. Empty frames teach the model what background looks like.

## Training (after review)

```bash
python split_dataset.py --reviewed-only     # holds out obstacle_course_trial2 for validation
yolo detect train data=dataset/data.yaml model=yolo11s.pt imgsz=1280 epochs=100 batch=8
```

## Data protection

Frames from `pedestrian` (and some from `house`) show an identifiable person. Treat the dataset as personal data under UK/EU GDPR: keep it internal, make sure there is consent or another lawful basis for using it, and blur faces before sharing it outside the team.
