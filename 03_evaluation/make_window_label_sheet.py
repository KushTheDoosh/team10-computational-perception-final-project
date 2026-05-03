#!/usr/bin/env python3

import csv
import sys


def prefill_labels(window_type):
    human = 1 if ('human_detected_window' in window_type or 'alert_lingering' in window_type) else 0
    suspicious = 1 if 'alert_lingering' in window_type else 0
    animal_abn = 0
    behavior = 'lingering' if 'alert_lingering' in window_type else ''
    return human, suspicious, animal_abn, behavior


def main(review_windows_csv, out_csv):
    with open(review_windows_csv, 'r', encoding='utf-8') as f:
        windows = list(csv.DictReader(f))

    fieldnames = [
        'window_id',
        'window_type',
        'start_frame',
        'end_frame',
        'start_sec',
        'end_sec',
        'human_present',
        'suspicious_event',
        'animal_abnormal_event',
        'behavior_label',
        'notes',
    ]

    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, row in enumerate(windows, start=1):
            human, suspicious, animal_abn, behavior = prefill_labels(row['window_type'])
            w.writerow(
                {
                    'window_id': i,
                    'window_type': row['window_type'],
                    'start_frame': row['start_frame'],
                    'end_frame': row['end_frame'],
                    'start_sec': row['start_sec'],
                    'end_sec': row['end_sec'],
                    'human_present': human,
                    'suspicious_event': suspicious,
                    'animal_abnormal_event': animal_abn,
                    'behavior_label': behavior,
                    'notes': row.get('notes', ''),
                }
            )

    print(f'Wrote window label sheet: {out_csv}')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: make_window_label_sheet.py <review_windows_csv> <out_csv>')
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
