#!/usr/bin/env python3

import argparse
import csv
import datetime
import json
import os
import time
from collections import defaultdict, deque

import cv2
import numpy as np


DEFAULT_CONFIG = {
    "FRAME_WIDTH": 640,
    "FRAME_HEIGHT": 480,
    "HOG_WIN_STRIDE": (8, 8),
    "HOG_PADDING": (4, 4),
    "HOG_SCALE": 1.05,
    "NMS_THRESHOLD": 0.45,
    "MOG2_HISTORY": 500,
    "MOG2_VAR_THRESH": 25,
    "MOG2_DETECT_SHADOW": True,
    "MORPH_KERNEL": (5, 5),
    "MIN_CONTOUR_AREA": 800,
    "IDLE_SPEED": 2.0,
    "WALKING_SPEED": 10.0,
    "LINGER_FRAMES": 90,
    "ANIMAL_STILL_FRAMES": 1800,
    "ALERT_COOLDOWN_S": 5.0,
}


def safe_div(num, den):
    return float(num) / float(den) if den else 0.0


class Preprocessor:
    def __init__(self, config):
        self.w = config["FRAME_WIDTH"]
        self.h = config["FRAME_HEIGHT"]

    def process(self, frame):
        resized = cv2.resize(frame, (self.w, self.h))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        equalized = cv2.equalizeHist(blurred)
        return {"resized": resized, "equalized": equalized}


class AnimalDetector:
    def __init__(self, config):
        self.config = config
        self.mog2 = cv2.createBackgroundSubtractorMOG2(
            history=config["MOG2_HISTORY"],
            varThreshold=config["MOG2_VAR_THRESH"],
            detectShadows=config["MOG2_DETECT_SHADOW"],
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config["MORPH_KERNEL"])

    def detect(self, frame):
        fg = self.mog2.apply(frame)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        fg = cv2.erode(fg, self.kernel, iterations=1)
        fg = cv2.dilate(fg, self.kernel, iterations=3)
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for cnt in contours:
            if cv2.contourArea(cnt) < self.config["MIN_CONTOUR_AREA"]:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / float(max(h, 1))
            if 0.5 < aspect < 4.0:
                boxes.append((x, y, w, h))

        return boxes


class HumanDetector:
    def __init__(self, config):
        self.config = config
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame):
        boxes, weights = self.hog.detectMultiScale(
            frame,
            winStride=self.config["HOG_WIN_STRIDE"],
            padding=self.config["HOG_PADDING"],
            scale=self.config["HOG_SCALE"],
        )
        if len(boxes) == 0:
            return [], []
        return self._nms(boxes, weights)

    def _nms(self, boxes, weights):
        boxes = np.array(boxes, dtype=np.float32)
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = weights.flatten().argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            order = order[np.where(iou <= self.config["NMS_THRESHOLD"])[0] + 1]

        keep_boxes = [tuple(boxes[k].astype(int)) for k in keep]
        keep_weights = [float(weights.flatten()[k]) for k in keep]
        return keep_boxes, keep_weights


class CentroidTracker:
    def __init__(self, max_disappeared=30):
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, obj_id):
        del self.objects[obj_id]
        del self.disappeared[obj_id]

    def update(self, boxes):
        if len(boxes) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return self.objects

        input_centroids = np.array([(x + w // 2, y + h // 2) for (x, y, w, h) in boxes])

        if len(self.objects) == 0:
            for c in input_centroids:
                self.register(c)
            return self.objects

        obj_ids = list(self.objects.keys())
        obj_centroids = list(self.objects.values())
        obj_arr = np.array(obj_centroids, dtype=np.float32)
        in_arr = np.array(input_centroids, dtype=np.float32)
        D = np.sqrt(((obj_arr[:, None, :] - in_arr[None, :, :]) ** 2).sum(axis=2))
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for (r, c) in zip(rows, cols):
            if r in used_rows or c in used_cols:
                continue
            obj_id = obj_ids[r]
            self.objects[obj_id] = input_centroids[c]
            self.disappeared[obj_id] = 0
            used_rows.add(r)
            used_cols.add(c)

        unused_rows = set(range(len(obj_centroids))) - used_rows
        unused_cols = set(range(len(input_centroids))) - used_cols

        for r in unused_rows:
            obj_id = obj_ids[r]
            self.disappeared[obj_id] += 1
            if self.disappeared[obj_id] > self.max_disappeared:
                self.deregister(obj_id)

        for c in unused_cols:
            self.register(input_centroids[c])

        return self.objects


class BehaviorClassifier:
    def __init__(self, config, history_len=30):
        self.config = config
        self.history = defaultdict(lambda: deque(maxlen=history_len))
        self.dwell_count = defaultdict(int)
        self.dwell_pos = {}
        self.dwell_radius = 25

    def classify(self, obj_id, centroid, entity_type="human"):
        cx, cy = centroid
        self.history[obj_id].append((cx, cy))

        if len(self.history[obj_id]) < 2:
            return "idle", 0.0

        prev = self.history[obj_id][-2]
        speed = float(np.sqrt((cx - prev[0]) ** 2 + (cy - prev[1]) ** 2))

        if obj_id not in self.dwell_pos:
            self.dwell_pos[obj_id] = (cx, cy)
        else:
            dpx, dpy = self.dwell_pos[obj_id]
            d = np.sqrt((cx - dpx) ** 2 + (cy - dpy) ** 2)
            if d < self.dwell_radius:
                self.dwell_count[obj_id] += 1
            else:
                self.dwell_count[obj_id] = 0
                self.dwell_pos[obj_id] = (cx, cy)

        dwell = self.dwell_count[obj_id]

        if entity_type == "animal":
            if dwell > self.config["ANIMAL_STILL_FRAMES"]:
                return "stationary_alert", speed
            if speed < self.config["IDLE_SPEED"]:
                return "grazing", speed
            return "moving", speed

        if speed > self.config["WALKING_SPEED"] and dwell > self.config["LINGER_FRAMES"] // 3:
            return "suspicious", speed
        if speed > self.config["WALKING_SPEED"]:
            return "running", speed
        if dwell > self.config["LINGER_FRAMES"]:
            return "lingering", speed
        if speed > self.config["IDLE_SPEED"]:
            return "walking", speed
        return "idle", speed


class AlertSystem:
    def __init__(self, cooldown_s=5.0):
        self.cooldown_s = cooldown_s
        self.last_fired = {}
        self.alerts = []

    def fire(self, frame_idx, alert_type, level, message):
        now = time.time()
        key = f"{alert_type}:{message}"
        if key in self.last_fired and (now - self.last_fired[key]) < self.cooldown_s:
            return False

        self.last_fired[key] = now
        self.alerts.append(
            {
                "frame_idx": int(frame_idx),
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "alert_type": alert_type,
                "level": level,
                "message": message,
            }
        )
        return True


def dominant_behavior(human_behaviors):
    if not human_behaviors:
        return "none"
    priority = ["suspicious", "lingering", "running", "walking", "idle"]
    for label in priority:
        if label in human_behaviors:
            return label
    return human_behaviors[0]


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_pipeline(video_path, out_dir, config, max_frames=None, write_video=False):
    os.makedirs(out_dir, exist_ok=True)

    pre = Preprocessor(config)
    ani = AnimalDetector(config)
    hum = HumanDetector(config)
    h_tracker = CentroidTracker(max_disappeared=25)
    a_tracker = CentroidTracker(max_disappeared=40)
    beh = BehaviorClassifier(config)
    alerts = AlertSystem(cooldown_s=config["ALERT_COOLDOWN_S"])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    in_fps = float(cap.get(cv2.CAP_PROP_FPS) or 20.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    writer = None
    out_video_path = ""
    if write_video:
        out_video_path = os.path.join(out_dir, "annotated_output.avi")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(
            out_video_path,
            fourcc,
            in_fps,
            (config["FRAME_WIDTH"], config["FRAME_HEIGHT"]),
        )

    frame_rows = []
    t0 = time.time()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames is not None and frame_idx >= max_frames:
            break

        pp = pre.process(frame)
        resized = pp["resized"]

        animal_boxes = ani.detect(resized)
        a_objects = a_tracker.update(animal_boxes)

        human_boxes, human_weights = hum.detect(pp["equalized"])
        h_objects = h_tracker.update(human_boxes)

        human_behaviors = []
        suspicious_alert = 0
        animal_abnormal_alert = 0
        alerts_before = len(alerts.alerts)

        for obj_id, centroid in h_objects.items():
            behavior, speed = beh.classify(obj_id, centroid, "human")
            human_behaviors.append(behavior)
            if behavior == "suspicious":
                fired = alerts.fire(
                    frame_idx,
                    alert_type="suspicious",
                    level="HIGH",
                    message=f"Suspicious human behavior ID={obj_id} speed={speed:.2f}",
                )
                if fired:
                    suspicious_alert = 1
            elif behavior == "lingering":
                alerts.fire(
                    frame_idx,
                    alert_type="lingering",
                    level="MEDIUM",
                    message=f"Human lingering ID={obj_id}",
                )

        for obj_id, centroid in a_objects.items():
            behavior, _ = beh.classify(obj_id, centroid, "animal")
            if behavior == "stationary_alert":
                fired = alerts.fire(
                    frame_idx,
                    alert_type="animal_stationary",
                    level="MEDIUM",
                    message=f"Animal stationary too long ID={obj_id}",
                )
                if fired:
                    animal_abnormal_alert = 1

        frame_alert_count = len(alerts.alerts) - alerts_before

        if writer is not None:
            vis = resized.copy()
            for (x, y, w, h), wt in zip(human_boxes, human_weights):
                cv2.rectangle(vis, (x, y), (x + w, y + h), (32, 160, 224), 2)
                cv2.putText(
                    vis,
                    f"HUMAN {wt:.2f}",
                    (x, max(0, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (32, 160, 224),
                    1,
                )
            for (x, y, w, h) in animal_boxes:
                cv2.rectangle(vis, (x, y), (x + w, y + h), (50, 200, 50), 2)
                cv2.putText(
                    vis,
                    "ANIMAL",
                    (x, max(0, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (50, 200, 50),
                    1,
                )
            writer.write(vis)

        frame_rows.append(
            {
                "frame_idx": frame_idx,
                "human_detected": int(len(human_boxes) > 0),
                "num_humans": len(human_boxes),
                "num_animals": len(animal_boxes),
                "primary_human_behavior": dominant_behavior(human_behaviors),
                "suspicious_alert": suspicious_alert,
                "animal_abnormal_alert": animal_abnormal_alert,
                "alerts_fired_count": frame_alert_count,
            }
        )

        frame_idx += 1

    elapsed = max(time.time() - t0, 1e-9)
    processed_frames = frame_idx
    proc_fps = processed_frames / elapsed

    cap.release()
    if writer is not None:
        writer.release()

    pred_csv = os.path.join(out_dir, "predictions_per_frame.csv")
    write_csv(
        pred_csv,
        fieldnames=[
            "frame_idx",
            "human_detected",
            "num_humans",
            "num_animals",
            "primary_human_behavior",
            "suspicious_alert",
            "animal_abnormal_alert",
            "alerts_fired_count",
        ],
        rows=frame_rows,
    )

    alerts_csv = os.path.join(out_dir, "alerts.csv")
    write_csv(
        alerts_csv,
        fieldnames=["frame_idx", "timestamp", "alert_type", "level", "message"],
        rows=alerts.alerts,
    )

    summary = {
        "video_path": video_path,
        "processed_frames": processed_frames,
        "source_fps": in_fps,
        "processing_fps": proc_fps,
        "avg_frame_time_ms": 1000.0 / proc_fps if proc_fps > 0 else 0.0,
        "duration_sec": processed_frames / in_fps if in_fps > 0 else 0.0,
        "total_alerts": len(alerts.alerts),
        "output_video": out_video_path,
        "predictions_csv": pred_csv,
        "alerts_csv": alerts_csv,
    }
    summary_json = os.path.join(out_dir, "run_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return frame_rows, alerts.alerts, summary


def load_gt_rows(gt_csv_path):
    gt = {}
    with open(gt_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"frame_idx", "human_present", "suspicious_event", "animal_abnormal_event"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing GT columns: {sorted(missing)}")

        for row in reader:
            frame_idx = int(row["frame_idx"])
            gt[frame_idx] = {
                "human_present": int(row.get("human_present", 0)),
                "suspicious_event": int(row.get("suspicious_event", 0)),
                "animal_abnormal_event": int(row.get("animal_abnormal_event", 0)),
                "behavior_label": (row.get("behavior_label", "") or "").strip().lower(),
            }
    return gt


def event_windows(series):
    windows = []
    start = None
    for idx, val in enumerate(series):
        if val == 1 and start is None:
            start = idx
        if val == 0 and start is not None:
            windows.append((start, idx - 1))
            start = None
    if start is not None:
        windows.append((start, len(series) - 1))
    return windows


def compute_alert_latency(gt_series, alert_frames, fps):
    windows = event_windows(gt_series)
    latencies = []
    missed = 0
    for start, end in windows:
        candidates = [f for f in alert_frames if start <= f <= end]
        if not candidates:
            missed += 1
            continue
        latencies.append((min(candidates) - start) / max(fps, 1e-9))

    return {
        "events": len(windows),
        "matched": len(latencies),
        "missed": missed,
        "avg_latency_sec": float(np.mean(latencies)) if latencies else None,
        "max_latency_sec": float(np.max(latencies)) if latencies else None,
    }


def compute_metrics(frame_rows, alerts, gt_rows, source_fps):
    max_frame = max(r["frame_idx"] for r in frame_rows) if frame_rows else -1

    human_tp = human_fp = human_fn = 0
    susp_tp = susp_fp = susp_fn = 0
    anim_tp = anim_fp = anim_fn = 0

    behavior_correct = 0
    behavior_total = 0
    behavior_confusion = defaultdict(lambda: defaultdict(int))

    gt_suspicious_series = []
    gt_animal_series = []

    for i in range(max_frame + 1):
        pred = frame_rows[i]
        gt = gt_rows.get(
            i,
            {
                "human_present": 0,
                "suspicious_event": 0,
                "animal_abnormal_event": 0,
                "behavior_label": "",
            },
        )

        p_h = int(pred["human_detected"])
        g_h = int(gt["human_present"])
        if p_h == 1 and g_h == 1:
            human_tp += 1
        elif p_h == 1 and g_h == 0:
            human_fp += 1
        elif p_h == 0 and g_h == 1:
            human_fn += 1

        p_s = int(pred["suspicious_alert"])
        g_s = int(gt["suspicious_event"])
        if p_s == 1 and g_s == 1:
            susp_tp += 1
        elif p_s == 1 and g_s == 0:
            susp_fp += 1
        elif p_s == 0 and g_s == 1:
            susp_fn += 1

        p_a = int(pred["animal_abnormal_alert"])
        g_a = int(gt["animal_abnormal_event"])
        if p_a == 1 and g_a == 1:
            anim_tp += 1
        elif p_a == 1 and g_a == 0:
            anim_fp += 1
        elif p_a == 0 and g_a == 1:
            anim_fn += 1

        gt_label = (gt.get("behavior_label") or "").strip().lower()
        if gt_label and gt_label != "none":
            behavior_total += 1
            pred_label = str(pred["primary_human_behavior"]).strip().lower()
            behavior_confusion[gt_label][pred_label] += 1
            if gt_label == pred_label:
                behavior_correct += 1

        gt_suspicious_series.append(g_s)
        gt_animal_series.append(g_a)

    human_precision = safe_div(human_tp, human_tp + human_fp)
    human_recall = safe_div(human_tp, human_tp + human_fn)
    human_f1 = safe_div(2 * human_precision * human_recall, human_precision + human_recall)

    suspicious_precision = safe_div(susp_tp, susp_tp + susp_fp)
    suspicious_recall = safe_div(susp_tp, susp_tp + susp_fn)

    animal_precision = safe_div(anim_tp, anim_tp + anim_fp)
    animal_recall = safe_div(anim_tp, anim_tp + anim_fn)

    behavior_accuracy = safe_div(behavior_correct, behavior_total)

    n_frames = len(frame_rows)
    hours = (n_frames / max(source_fps, 1e-9)) / 3600.0

    false_alerts = 0
    for a in alerts:
        frame_idx = int(a["frame_idx"])
        gt = gt_rows.get(
            frame_idx,
            {"suspicious_event": 0, "animal_abnormal_event": 0},
        )
        if int(gt.get("suspicious_event", 0)) == 0 and int(gt.get("animal_abnormal_event", 0)) == 0:
            false_alerts += 1

    far_per_hour = safe_div(false_alerts, hours)

    suspicious_alert_frames = [int(a["frame_idx"]) for a in alerts if a["alert_type"] == "suspicious"]
    animal_alert_frames = [int(a["frame_idx"]) for a in alerts if a["alert_type"] == "animal_stationary"]

    latency_suspicious = compute_alert_latency(gt_suspicious_series, suspicious_alert_frames, source_fps)
    latency_animal = compute_alert_latency(gt_animal_series, animal_alert_frames, source_fps)

    metrics = {
        "human_detection": {
            "precision": human_precision,
            "recall": human_recall,
            "f1": human_f1,
            "tp": human_tp,
            "fp": human_fp,
            "fn": human_fn,
        },
        "suspicious_alert_frame_metrics": {
            "precision": suspicious_precision,
            "recall": suspicious_recall,
            "tp": susp_tp,
            "fp": susp_fp,
            "fn": susp_fn,
        },
        "animal_abnormal_alert_frame_metrics": {
            "precision": animal_precision,
            "recall": animal_recall,
            "tp": anim_tp,
            "fp": anim_fp,
            "fn": anim_fn,
        },
        "behavior_classification": {
            "accuracy": behavior_accuracy,
            "labeled_frames": behavior_total,
            "correct": behavior_correct,
            "confusion": {gt: dict(preds) for gt, preds in behavior_confusion.items()},
        },
        "false_alarm_rate_per_hour": far_per_hour,
        "false_alert_count": false_alerts,
        "duration_hours": hours,
        "alert_latency": {
            "suspicious": latency_suspicious,
            "animal_abnormal": latency_animal,
        },
    }

    return metrics


def create_gt_template(frame_rows, out_path):
    rows = []
    for r in frame_rows:
        rows.append(
            {
                "frame_idx": r["frame_idx"],
                "human_present": 0,
                "suspicious_event": 0,
                "animal_abnormal_event": 0,
                "behavior_label": "",
            }
        )

    write_csv(
        out_path,
        fieldnames=[
            "frame_idx",
            "human_present",
            "suspicious_event",
            "animal_abnormal_event",
            "behavior_label",
        ],
        rows=rows,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Team 10 surveillance evaluation runner")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--output-dir", required=True, help="Directory for outputs")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame cap")
    parser.add_argument("--write-video", action="store_true", help="Write annotated output video")
    parser.add_argument("--gt-csv", default=None, help="Optional GT CSV for metrics")
    parser.add_argument(
        "--make-gt-template",
        action="store_true",
        help="Create gt_template.csv from processed frame indices",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = DEFAULT_CONFIG.copy()

    frame_rows, alerts, summary = run_pipeline(
        video_path=args.video,
        out_dir=args.output_dir,
        config=config,
        max_frames=args.max_frames,
        write_video=args.write_video,
    )

    print(f"Processed frames: {summary['processed_frames']}")
    print(f"Source FPS: {summary['source_fps']:.2f}")
    print(f"Processing FPS: {summary['processing_fps']:.2f}")
    print(f"Total alerts: {summary['total_alerts']}")
    print(f"Predictions CSV: {summary['predictions_csv']}")
    print(f"Alerts CSV: {summary['alerts_csv']}")

    if args.make_gt_template:
        gt_template = os.path.join(args.output_dir, "gt_template.csv")
        create_gt_template(frame_rows, gt_template)
        print(f"GT template created: {gt_template}")

    if args.gt_csv:
        gt_rows = load_gt_rows(args.gt_csv)
        metrics = compute_metrics(frame_rows, alerts, gt_rows, summary["source_fps"])
        metrics_path = os.path.join(args.output_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        print(f"Metrics JSON: {metrics_path}")
        print("Human precision/recall/F1:",
              f"{metrics['human_detection']['precision']:.4f}/"
              f"{metrics['human_detection']['recall']:.4f}/"
              f"{metrics['human_detection']['f1']:.4f}")
        print("Behavior accuracy:", f"{metrics['behavior_classification']['accuracy']:.4f}")
        print("False alarms/hour:", f"{metrics['false_alarm_rate_per_hour']:.3f}")


if __name__ == "__main__":
    main()
