# Intelligent Surveillance for Theft Detection and Animal Behavior Analytics

> Final draft for Team 10 writeup handoff (ComS 5750)

## 1. Abstract
This project presents a real-time surveillance pipeline that jointly performs (1) human-presence monitoring for theft-related risk and (2) abnormal animal-behavior monitoring from a single video stream. The implemented system combines classical computer-vision components: HOG+SVM human detection, MOG2 foreground segmentation, contour-based animal localization, temporal tracking, and rule-based behavior classification. Primary quantitative evaluation was conducted on 28 manually reviewed clips (3 seconds each) extracted from two horse-stall videos (273.96 seconds total). At the default clip-level human rule (`human_detected_ratio >= 0.12`), human-presence performance was 28.6% precision, 20.0% recall, and 23.5% F1. A recall-first threshold (`>= 0.00`) improved human recall to 100.0% with 52.6% precision (F1 69.0%). Runtime remained real-time at 51.89 FPS (19.27 ms/frame). Additional short validation videos were used for qualitative and runtime checks only, not primary precision/recall claims.

## 2. Introduction
Conventional surveillance systems are largely passive: they record footage but require manual review after incidents occur. This delay is costly in settings where immediate intervention matters, such as theft prevention in storage areas and early response to abnormal livestock behavior. The key gap is that many systems detect motion but do not interpret behavior over time.

This project addresses that gap by building an end-to-end computational perception pipeline that converts raw video frames into actionable alerts. Instead of stopping at object detection, the system estimates temporal movement features and classifies behavior into operational categories and alert logic suitable for live monitoring.

### 2.1 Contributions
1. A unified dual-pipeline architecture for human monitoring and animal monitoring from a shared video stream.
2. A temporal behavior classification layer that converts movement trajectories into interpretable behavioral labels.
3. A reproducible clip-level evaluation protocol with manual review labels.
4. A live monitoring interface with real-time annotation and alerting.

## 3. Project Scope and Problem Definition
### In scope
- Real-time person detection for potential theft-related events.
- Animal movement monitoring and abnormal behavior flagging.
- Alert generation from temporal behavior signals.
- End-to-end demonstration in a monitoring dashboard.

### Out of scope
- Fine-grained species identification across diverse wildlife.
- Multi-camera identity handoff across non-overlapping views.
- Fully learned end-to-end anomaly detection requiring large-scale labeled video datasets.

## 4. Related Work
Prior surveillance systems establish strong baselines for frame-level detection, but fewer systems provide lightweight behavior interpretation suitable for real-time deployment without specialized hardware. HOG+SVM remains a practical classical baseline for person detection, and MOG2 remains widely used for foreground extraction in constrained scenes. Recent deep-learning literature improves robustness in occlusion and low-light conditions at higher compute cost.

### 4.1 Literature Expansion Target
For final submission, include at least 15 references (or your section minimum), including:
1. Classical detection foundations (HOG/SVM, background subtraction).
2. Modern detectors (YOLO-family or equivalent) for comparative discussion.
3. Multi-object tracking (SORT/DeepSORT/ByteTrack) for temporal consistency.
4. Video anomaly detection and behavior-analysis papers.
5. Domain-specific animal monitoring and smart-farming studies.

## 5. System Methodology
### 5.1 Pipeline Overview
Each frame passes through five stages:
1. Video capture and preprocessing.
2. Human detection.
3. Animal foreground detection.
4. Temporal behavior classification.
5. Alert generation and visualization.

### 5.2 Stage 1: Video Capture and Preprocessing
Frames are resized to fixed resolution and converted to grayscale/equalized views for detector stability.

### 5.3 Stage 2: Human Detection
Human candidates are detected with HOG descriptors and a linear SVM via multi-scale sliding windows, followed by NMS.

### 5.4 Stage 3: Animal Foreground Detection
MOG2 foreground masks, morphological cleanup, and contour area filtering are used to localize animals.

### 5.5 Stage 4: Behavior Classification
Tracked entities are assigned rule-based labels from short-term movement features (speed, dwell, direction change).

### 5.6 Stage 5: Alerting and Interface
Alerts are emitted for suspicious/lingering human behavior and prolonged animal-stationary states, then visualized with frame overlays and logs.

## 6. Experimental Setup
### 6.1 Hardware and Runtime
- Device: MacBook Air (Apple M4, 10-core CPU; arm64).
- RAM: 16 GB (17,179,869,184 bytes).
- OS: macOS 26.3.1 (Build 25D771280a).
- Software: Python 3.14.3, OpenCV 4.13.0.

### 6.2 Data Sources
Primary evaluated set:
- `horseVId.mp4` (56.97 s): baseline horse-stall footage with no human interaction events.
- `horsevid2.mp4` (216.99 s): horse-stall footage including veterinary interaction segments plus horse-only periods.
- Clip protocol: 28 non-overlapping clips of 3 seconds each (10 from `horseVId.mp4`, 18 from `horsevid2.mp4`) with manual clip-level labels.

Additional validation/runtime examples:
- `human.mp4` (5.88 s)
- `animal.mp4` (5.88 s)
These were used only for qualitative sanity checks and throughput reporting, not for the primary precision/recall claims.

### 6.3 Test Conditions
Primary labeled evaluation conditions:
- E1 No-human baseline clips.
- E2 Human-present intervention clips.
- E3 Animal-handling clips labeled as abnormal events.

### 6.4 Metrics
- Detection: precision, recall, F1, and accuracy (clip-level).
- Behavior: label accuracy on reviewed clips.
- Alert quality: false alarms per hour.
- Runtime: FPS and frame processing time.

## 7. Results
### 7.1 Quantitative Summary (Primary Labeled Horse Set)
| Metric | Observed |
|---|---:|
| Human precision (default threshold) | 28.6% |
| Human recall (default threshold) | 20.0% |
| Human F1 (default threshold) | 23.5% |
| Human precision (recall-first tuned threshold) | 52.6% |
| Human recall (recall-first tuned threshold) | 100.0% |
| Human F1 (recall-first tuned threshold) | 69.0% |
| Behavior accuracy (human-labeled clips) | 20.0% (2/10) |
| False alarms/hour (overall) | 52.56/hr |
| Throughput | 51.89 FPS |

### 7.2 Per-Source Results (Primary Labeled Horse Set)
| Source | Clips | Human Precision | Human Recall | Human F1 | Behavior Accuracy | FAR/hr | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| `horseVId.mp4` (baseline, no human positives) | 10 | N/A | N/A | N/A | N/A | 0.00 | 55.54 |
| `horsevid2.mp4` (human interaction segments) | 18 | 0.286 | 0.200 | 0.235 | 0.200 (2/10) | 66.36 | 50.93 |
| Combined (default threshold) | 28 | 0.286 | 0.200 | 0.235 | 0.200 (2/10) | 52.56 | 51.89 |
| Combined (tuned threshold) | 28 | 0.526 | 1.000 | 0.690 | 0.200 (2/10) | 52.56 | 51.89 |

### 7.3 Additional Validation and Runtime Examples (New Videos)
These two videos are intentionally reported as qualitative/runtime validation only due their short combined duration (11.75 s) and lack of ground-truth annotation.

| Video | Frames | Duration (s) | Processing FPS | Total Alerts | Human-detected Frames |
|---|---:|---:|---:|---:|---:|
| `human.mp4` | 141 | 5.88 | 44.55 | 0 | 141 (100.0%) |
| `animal.mp4` | 141 | 5.88 | 44.81 | 0 | 123 (87.2%) |
| Combined | 282 | 11.75 | 44.68 (weighted) | 0 | 264 (93.6%) |

### 7.4 Key Findings
1. Runtime is consistently real-time on laptop hardware (44.68 to 51.89 FPS depending on input).
2. Default human sensitivity on primary labeled clips is conservative (recall 20.0%).
3. Recall-first thresholding improves coverage (100.0% recall) at lower precision (52.6%).
4. Animal-abnormal detection remains weak on labeled positives in the primary set and needs redesign.

## 8. Error Analysis and Discussion
Main failure patterns observed:
1. Human false negatives during brief/partial-occlusion intervention moments.
2. Human false positives from stall scene motion/shape artifacts.
3. Animal-abnormal misses where handling events did not cross motion-only thresholds.
4. Boundary mismatch between frame-level logic and 3-second clip-level labels.

Mitigations:
1. Calibrate thresholds on a held-out validation subset.
2. Compare against a lightweight modern detector (e.g., YOLOv8n person class).
3. Add contextual abnormal cues (interaction proximity + duration).
4. Log event-level latency directly.

## 9. Limitations and Future Work
This is a constrained proof-of-concept evaluation. The primary labeled dataset is small (28 clips) and class-imbalanced (0 suspicious positives, 2 animal-abnormal positives), so uncertainty is high. The additional short videos strengthen runtime evidence but do not replace labeled quantitative evaluation.

Future work:
1. Build a larger balanced labeled dataset with sufficient positives per event type.
2. Upgrade person detection/tracking robustness.
3. Add multi-reviewer label agreement and adjudication.
4. Report event-level metrics (time-to-detect/time-to-alert).

## 10. Conclusion
The project delivers an end-to-end computational perception pipeline for human-presence and animal-behavior surveillance with strong real-time performance. Primary labeled results reveal meaningful tradeoffs between sensitivity and precision, and expose clear improvement targets in abnormal-event detection. The system is suitable as a course-level working prototype with transparent limitations and a clear roadmap for improvement.

## 11. Final Handoff Checklist
- [x] Main results anchored to manually labeled horse evaluation.
- [x] New videos included as additional qualitative/runtime validation.
- [x] Placeholder metrics removed.
- [x] Method and limitations aligned with actual pipeline behavior.
- [ ] Add final figures (architecture + sample detections + metrics plot).
- [ ] Expand references to final required count.
- [ ] Do one final grammar/style pass before submission.
