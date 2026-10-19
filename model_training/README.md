# Model training: YOLOv8s fine-tune

This trains on the labels in `../annotation/dataset`. Only images marked **reviewed** in the annotation tool are used by default.

## Run

```bash
cd model_training
python train.py                 # reviewed images only
python train.py --all           # also use unreviewed auto-labels (lower quality)
python train.py --batch 4       # if you hit CUDA out-of-memory
python train.py --resume        # continue an interrupted run
```

**Windows "WinError 1455 / paging file too small":** each data-loader worker loads its own copy of torch/CUDA, which exhausts the Windows page file. That's why the default on Windows is `--workers 0` with a disk image cache. For faster training, enlarge the page file (System → Advanced system settings → Performance → Virtual memory: set a custom size, e.g. 16384–32768 MB, then reboot) and use `--workers 2`.

Defaults: `yolov8s.pt`, 1280 px images (needed for small distant objects), 100 epochs, early stop after 30 epochs without improvement, batch 8.

**Validation:** the whole `obstacle_course_trial2` video is held out. Neighbouring frames are almost identical, so a random split would give inflated scores. If that video has no reviewed frames, the script uses the last 15% of each video instead. Change the held-out video(s) with `--val-videos pedestrian`.

## Outputs

- `splits/`: train/val lists and `data.yaml`, regenerated on every run
- `runs/yolov8s_ft/`: training curves (`results.png`), confusion matrix and example predictions on the validation set
- `weights/best.pt`: the best checkpoint

## Check it on the videos

```bash
python predict_video.py                 # writes predictions/*_pred.mp4
python predict_video.py --conf 0.35     # fewer false positives
```

## Notes

- 575 frames from 5 short clips is a small dataset. Expect good results on these clips and weaker results on new locations, lighting or props. Adding more varied footage is what will improve it.
- Classes with few boxes (barrier, house) will be the least reliable. Check their per-class scores in the final validation printout.
- Frames that show a person are personal data under UK/EU GDPR, so keep `runs/` and `predictions/` internal.
