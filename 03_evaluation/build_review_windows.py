#!/usr/bin/env python3

import csv
import os
import sys


def read_rows(path):
    with open(path, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def contiguous_windows(frame_indices):
    if not frame_indices:
        return []
    frame_indices = sorted(frame_indices)
    windows = []
    start = frame_indices[0]
    prev = frame_indices[0]

    for idx in frame_indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        windows.append((start, prev))
        start = idx
        prev = idx

    windows.append((start, prev))
    return windows


def write_windows(out_path, window_rows):
    fieldnames = [
        'window_type',
        'start_frame',
        'end_frame',
        'length_frames',
        'start_sec',
        'end_sec',
        'notes',
    ]
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in window_rows:
            writer.writerow(row)


def main(pred_csv, alerts_csv, out_csv, fps=25.0):
    preds = read_rows(pred_csv)
    alerts = read_rows(alerts_csv)

    human_frames = [int(r['frame_idx']) for r in preds if int(r['human_detected']) == 1]
    animal_frames = [int(r['frame_idx']) for r in preds if int(r['num_animals']) > 0]

    human_windows = contiguous_windows(human_frames)
    animal_windows = contiguous_windows(animal_frames)

    rows = []

    for s, e in human_windows:
        rows.append(
            {
                'window_type': 'human_detected_window',
                'start_frame': s,
                'end_frame': e,
                'length_frames': e - s + 1,
                'start_sec': round(s / fps, 2),
                'end_sec': round(e / fps, 2),
                'notes': 'Verify human presence + behavior labels in this window',
            }
        )

    for s, e in animal_windows:
        rows.append(
            {
                'window_type': 'animal_motion_window',
                'start_frame': s,
                'end_frame': e,
                'length_frames': e - s + 1,
                'start_sec': round(s / fps, 2),
                'end_sec': round(e / fps, 2),
                'notes': 'Verify animal abnormal-event GT if needed',
            }
        )

    for a in alerts:
        fr = int(a['frame_idx'])
        rows.append(
            {
                'window_type': f"alert_{a['alert_type']}",
                'start_frame': max(0, fr - 50),
                'end_frame': fr + 50,
                'length_frames': 101,
                'start_sec': round(max(0, fr - 50) / fps, 2),
                'end_sec': round((fr + 50) / fps, 2),
                'notes': a['message'],
            }
        )

    rows.sort(key=lambda r: (int(r['start_frame']), r['window_type']))
    write_windows(out_csv, rows)

    print(f'Wrote {len(rows)} windows to: {out_csv}')


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Usage: build_review_windows.py <predictions_csv> <alerts_csv> <out_csv> [fps]')
        sys.exit(1)
    pred_csv = sys.argv[1]
    alerts_csv = sys.argv[2]
    out_csv = sys.argv[3]
    fps = float(sys.argv[4]) if len(sys.argv) > 4 else 25.0
    main(pred_csv, alerts_csv, out_csv, fps)
