#!/usr/bin/env python3

import argparse
import csv
import os
from collections import Counter

import cv2


def read_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sample_even(seq, k):
    if not seq or k <= 0:
        return []
    if len(seq) <= k:
        return list(seq)
    out = []
    n = len(seq)
    for i in range(k):
        idx = round(i * (n - 1) / (k - 1))
        out.append(seq[idx])
    dedup = []
    seen = set()
    for x in out:
        if x in seen:
            continue
        seen.add(x)
        dedup.append(x)
    return dedup


def enforce_center_gap(centers, min_gap):
    if not centers:
        return []
    centers = sorted(centers)
    kept = [centers[0]]
    for c in centers[1:]:
        if c - kept[-1] >= min_gap:
            kept.append(c)
    return kept


def sec(frame_idx, fps):
    return round(frame_idx / fps, 3)


def build_window_summary(pred_rows, start_frame, end_frame):
    window = pred_rows[start_frame : end_frame + 1]
    if not window:
        return {
            "model_hint_human_any": 0,
            "model_hint_human_ratio": 0.0,
            "model_hint_max_humans": 0,
            "model_hint_max_animals": 0,
            "model_hint_top_behavior": "none",
            "model_hint_alert_frames": 0,
        }

    human_flags = [int(r["human_detected"]) for r in window]
    human_ratio = sum(human_flags) / len(human_flags)
    max_humans = max(int(r["num_humans"]) for r in window)
    max_animals = max(int(r["num_animals"]) for r in window)
    beh_counter = Counter((r.get("primary_human_behavior") or "none") for r in window)
    top_behavior = beh_counter.most_common(1)[0][0] if beh_counter else "none"
    alert_frames = sum(int(r.get("alerts_fired_count", 0)) > 0 for r in window)

    return {
        "model_hint_human_any": int(any(human_flags)),
        "model_hint_human_ratio": round(human_ratio, 3),
        "model_hint_max_humans": max_humans,
        "model_hint_max_animals": max_animals,
        "model_hint_top_behavior": top_behavior,
        "model_hint_alert_frames": int(alert_frames),
    }


def export_clip(cap, out_path, start_frame, end_frame, fps, width, height):
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    frame_idx = start_frame
    while frame_idx <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        frame_idx += 1
    writer.release()


def build_candidates(pred_rows, alert_rows, clip_frames, alert_target, pos_target, neg_target):
    alert_frames = sorted(int(a["frame_idx"]) for a in alert_rows)
    grouped_alert_frames = enforce_center_gap(alert_frames, min_gap=clip_frames)
    alert_centers = sample_even(grouped_alert_frames, alert_target)

    alert_frame_set = set(alert_frames)
    buffer_frames = clip_frames

    human_non_alert = []
    no_human = []

    for r in pred_rows:
        f = int(r["frame_idx"])
        h = int(r["human_detected"])
        near_alert = any(abs(f - af) <= buffer_frames for af in alert_frame_set) if alert_frame_set else False

        if h == 1 and not near_alert:
            human_non_alert.append(f)
        if h == 0 and int(r.get("alerts_fired_count", 0)) == 0:
            no_human.append(f)

    human_non_alert = enforce_center_gap(human_non_alert, min_gap=clip_frames)
    no_human = enforce_center_gap(no_human, min_gap=clip_frames)

    pos_centers = sample_even(human_non_alert, pos_target)
    neg_centers = sample_even(no_human, neg_target)

    candidates = []
    for c in alert_centers:
        candidates.append((c, "alert"))
    for c in pos_centers:
        candidates.append((c, "human_check"))
    for c in neg_centers:
        candidates.append((c, "negative_check"))

    # final dedup with preference order by category, then frame
    pref = {"alert": 0, "human_check": 1, "negative_check": 2}
    candidates = sorted(candidates, key=lambda x: (x[0], pref[x[1]]))

    merged = []
    for center, ctype in candidates:
        if not merged:
            merged.append((center, ctype))
            continue
        last_center, _ = merged[-1]
        if center - last_center >= clip_frames:
            merged.append((center, ctype))

    return merged


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    clips_dir = os.path.join(args.output_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)

    pred_rows = read_csv(args.predictions_csv)
    alert_rows = read_csv(args.alerts_csv)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or args.fps_fallback)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    clip_frames = max(1, round(args.clip_seconds * fps))
    half = clip_frames // 2

    candidates = build_candidates(
        pred_rows,
        alert_rows,
        clip_frames=clip_frames,
        alert_target=args.alert_target,
        pos_target=args.pos_target,
        neg_target=args.neg_target,
    )

    out_rows = []

    for idx, (center, ctype) in enumerate(candidates, start=1):
        start = max(0, center - half)
        end = min(total_frames - 1, start + clip_frames - 1)
        start = max(0, end - clip_frames + 1)

        win_summary = build_window_summary(pred_rows, start, end)

        related_alerts = [a for a in alert_rows if start <= int(a["frame_idx"]) <= end]
        alert_types = sorted(set(a["alert_type"] for a in related_alerts))

        clip_name = f"clip_{idx:04d}_{ctype}_f{start}_to_f{end}.mp4"
        clip_path = os.path.join(clips_dir, clip_name)

        export_clip(cap, clip_path, start, end, fps, width, height)

        out_rows.append(
            {
                "clip_id": idx,
                "clip_name": clip_name,
                "clip_path": clip_path,
                "clip_type": ctype,
                "center_frame": center,
                "start_frame": start,
                "end_frame": end,
                "start_sec": sec(start, fps),
                "end_sec": sec(end, fps),
                "duration_sec": round((end - start + 1) / fps, 3),
                "model_hint_human_any": win_summary["model_hint_human_any"],
                "model_hint_human_ratio": win_summary["model_hint_human_ratio"],
                "model_hint_max_humans": win_summary["model_hint_max_humans"],
                "model_hint_max_animals": win_summary["model_hint_max_animals"],
                "model_hint_top_behavior": win_summary["model_hint_top_behavior"],
                "model_hint_alert_frames": win_summary["model_hint_alert_frames"],
                "model_hint_alert_types": "|".join(alert_types),
                "human_present": "",
                "suspicious_event": "",
                "animal_abnormal_event": "",
                "behavior_label": "",
                "review_status": "",
                "reviewer_notes": "",
            }
        )

    cap.release()

    out_csv = os.path.join(args.output_dir, "review_sheet_precise.csv")
    fieldnames = [
        "clip_id",
        "clip_name",
        "clip_path",
        "clip_type",
        "center_frame",
        "start_frame",
        "end_frame",
        "start_sec",
        "end_sec",
        "duration_sec",
        "model_hint_human_any",
        "model_hint_human_ratio",
        "model_hint_max_humans",
        "model_hint_max_animals",
        "model_hint_top_behavior",
        "model_hint_alert_frames",
        "model_hint_alert_types",
        "human_present",
        "suspicious_event",
        "animal_abnormal_event",
        "behavior_label",
        "review_status",
        "reviewer_notes",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    meta_path = os.path.join(args.output_dir, "review_pack_meta.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"video={args.video}\n")
        f.write(f"fps={fps}\n")
        f.write(f"total_frames={total_frames}\n")
        f.write(f"clip_seconds={args.clip_seconds}\n")
        f.write(f"clip_frames={clip_frames}\n")
        f.write(f"total_clips={len(out_rows)}\n")
        f.write(f"review_csv={out_csv}\n")
        f.write(f"clips_dir={clips_dir}\n")

    print(f"Generated precise review pack")
    print(f"  Clips: {len(out_rows)}")
    print(f"  CSV  : {out_csv}")
    print(f"  Dir  : {clips_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Generate precise clip-review pack with 1:1 CSV mapping")
    p.add_argument("--video", required=True)
    p.add_argument("--predictions-csv", required=True)
    p.add_argument("--alerts-csv", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--clip-seconds", type=float, default=3.0)
    p.add_argument("--alert-target", type=int, default=30)
    p.add_argument("--pos-target", type=int, default=20)
    p.add_argument("--neg-target", type=int, default=20)
    p.add_argument("--fps-fallback", type=float, default=30.0)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
