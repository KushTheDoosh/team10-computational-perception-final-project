#!/usr/bin/env python3

import csv
import os
import sys

import cv2


def sanitize(name):
    return ''.join(ch if ch.isalnum() or ch in ('_', '-') else '_' for ch in name)


def main(video_path, windows_csv, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    with open(windows_csv, 'r', encoding='utf-8') as f:
        windows = list(csv.DictReader(f))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {video_path}')

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    for i, w in enumerate(windows, start=1):
        wtype = w['window_type']
        start = int(w['start_frame'])
        end = int(w['end_frame'])
        if end < start:
            continue

        filename = f"{i:03d}_{sanitize(wtype)}_{start}_{end}.mp4"
        out_path = os.path.join(out_dir, filename)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        writer = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height),
        )

        frame_idx = start
        while frame_idx <= end:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            frame_idx += 1

        writer.release()

    cap.release()
    print(f'Exported {len(windows)} clips to: {out_dir}')


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: export_window_clips.py <video_path> <review_windows_csv> <output_dir>')
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
