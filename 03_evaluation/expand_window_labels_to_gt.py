#!/usr/bin/env python3

import csv
import sys


def to_int(v, default=0):
    try:
        return int(v)
    except Exception:
        return default


def main(predictions_csv, window_labels_csv, out_gt_csv):
    with open(predictions_csv, 'r', encoding='utf-8') as f:
        preds = list(csv.DictReader(f))

    max_frame = max(int(r['frame_idx']) for r in preds)

    gt = [
        {
            'frame_idx': i,
            'human_present': 0,
            'suspicious_event': 0,
            'animal_abnormal_event': 0,
            'behavior_label': '',
        }
        for i in range(max_frame + 1)
    ]

    with open(window_labels_csv, 'r', encoding='utf-8') as f:
        windows = list(csv.DictReader(f))

    for w in windows:
        start = to_int(w.get('start_frame', 0), 0)
        end = to_int(w.get('end_frame', -1), -1)
        if end < start:
            continue

        human = to_int(w.get('human_present', 0), 0)
        suspicious = to_int(w.get('suspicious_event', 0), 0)
        animal_abn = to_int(w.get('animal_abnormal_event', 0), 0)
        behavior = (w.get('behavior_label', '') or '').strip().lower()

        start = max(0, start)
        end = min(max_frame, end)

        for i in range(start, end + 1):
            gt[i]['human_present'] = max(gt[i]['human_present'], human)
            gt[i]['suspicious_event'] = max(gt[i]['suspicious_event'], suspicious)
            gt[i]['animal_abnormal_event'] = max(gt[i]['animal_abnormal_event'], animal_abn)
            if behavior:
                gt[i]['behavior_label'] = behavior

    with open(out_gt_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                'frame_idx',
                'human_present',
                'suspicious_event',
                'animal_abnormal_event',
                'behavior_label',
            ],
        )
        w.writeheader()
        for row in gt:
            w.writerow(row)

    print(f'Wrote GT CSV: {out_gt_csv}')


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: expand_window_labels_to_gt.py <predictions_csv> <window_labels_csv> <out_gt_csv>')
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
