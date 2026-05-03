# Team 10 Evaluation Workflow

This folder lets you run your surveillance pipeline and compute report metrics with local files (no Colab required).

## Files
- `team10_eval_runner.py`: runs pipeline and exports predictions/alerts; computes metrics when GT is provided.
- `build_review_windows.py`: creates compact frame windows to speed up manual GT labeling.
- `export_window_clips.py`: exports short video clips for each review window.
- `make_window_label_sheet.py`: creates a compact window-level labeling sheet.
- `expand_window_labels_to_gt.py`: expands window labels into per-frame GT CSV.

## 1) Run the pipeline on a video
```bash
python3 team10_eval_runner.py \
  --video "/ABS/PATH/TO/VIDEO.mp4" \
  --output-dir "/ABS/PATH/TO/output_run" \
  --make-gt-template
```

Optional flags:
- `--max-frames 2000` to run only part of a video.
- `--write-video` to save annotated output video.

Outputs:
- `predictions_per_frame.csv`
- `alerts.csv`
- `run_summary.json`
- `gt_template.csv` (if `--make-gt-template` is used)

## 2) Generate review windows (faster labeling)
```bash
python3 build_review_windows.py \
  "/ABS/PATH/TO/predictions_per_frame.csv" \
  "/ABS/PATH/TO/alerts.csv" \
  "/ABS/PATH/TO/review_windows.csv" \
  25
```

Use `review_windows.csv` to focus GT labeling on meaningful segments first.

## 3) Fill `gt_template.csv`
Required columns:
- `frame_idx`
- `human_present` (0/1)
- `suspicious_event` (0/1)
- `animal_abnormal_event` (0/1)
- `behavior_label` (optional, e.g., `idle`, `walking`, `lingering`, `suspicious`, `running`)

Optional: export review-window clips first:
```bash
python3 export_window_clips.py \
  "/ABS/PATH/TO/VIDEO.mp4" \
  "/ABS/PATH/TO/review_windows.csv" \
  "/ABS/PATH/TO/review_clips"
```

Faster option (label windows instead of every frame):
```bash
python3 make_window_label_sheet.py \
  "/ABS/PATH/TO/review_windows.csv" \
  "/ABS/PATH/TO/window_label_sheet.csv"
```

After filling `window_label_sheet.csv`, expand to per-frame GT:
```bash
python3 expand_window_labels_to_gt.py \
  "/ABS/PATH/TO/predictions_per_frame.csv" \
  "/ABS/PATH/TO/window_label_sheet.csv" \
  "/ABS/PATH/TO/gt_from_windows.csv"
```

## 4) Compute metrics
```bash
python3 team10_eval_runner.py \
  --video "/ABS/PATH/TO/VIDEO.mp4" \
  --output-dir "/ABS/PATH/TO/output_run" \
  --gt-csv "/ABS/PATH/TO/gt_template_filled.csv"
```

This writes `metrics.json` including:
- human detection precision/recall/F1
- suspicious alert precision/recall
- animal abnormal alert precision/recall
- behavior accuracy + confusion map
- false alarm rate per hour
- alert latency stats
