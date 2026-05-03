# Intelligent Surveillance for Theft Detection and Animal Behavior Analytics

**Team 10 Final Project Draft (Submission-Ready Merge)**  
ComS 5750: Computational Perception  
Iowa State University

## Abstract
This report presents a real-time surveillance pipeline for (1) human-presence monitoring related to theft risk and (2) abnormal animal-behavior monitoring from a single video stream. The implemented system combines classical computer-vision components: HOG+SVM human detection, MOG2 foreground segmentation, contour-based animal localization, temporal tracking, and rule-based behavior classification. Primary quantitative evaluation was conducted on 28 manually reviewed clips (3 seconds each) from two horse-stall videos (273.96 seconds total). At the default clip-level human rule (`human_detected_ratio >= 0.12`), human-presence performance was 28.6% precision, 20.0% recall, and 23.5% F1. A recall-first threshold (`>= 0.00`) improved human recall to 100.0% with 52.6% precision (F1 69.0%). Runtime remained real-time at 51.89 FPS (19.27 ms/frame). Two additional short videos were used as qualitative/runtime validation only.

## 1. Introduction
Modern surveillance cameras record video but often lack actionable intelligence for timely intervention. In farms and warehouse settings, delayed review can reduce response effectiveness and increase operational risk [1], [17]. This project focuses on building a practical CPU-friendly pipeline that adds temporal behavior reasoning on top of frame-level detection.

### 1.1 Motivation and Target Users
Target users include:
- Farm owners and livestock managers who need continuous animal monitoring and intruder alerts.
- Warehouse security teams who need automated flagging of suspicious human presence.
- Wildlife/reserve operators who need scalable remote monitoring with limited personnel.

### 1.2 Objectives
1. Build a dual-channel pipeline for simultaneous human and animal monitoring from one feed.
2. Implement temporal behavior classification based on movement history.
3. Maintain real-time performance on standard laptop hardware.
4. Evaluate using manually reviewed labels and transparent failure analysis.

## 2. Related Work
### 2.1 Human Detection (HOG + SVM)
HOG-based pedestrian detection remains a practical baseline in low-resource settings [2], [8], [12]. Its strengths are interpretability and CPU efficiency, while limitations include sensitivity to occlusion and scale.

### 2.2 Foreground Segmentation and Animal Localization
Adaptive background subtraction methods (e.g., MOG2) are established for real-time foreground extraction [3], [13], [14]. In controlled camera views, contour filtering yields lightweight animal localization.

### 2.3 Temporal Behavioral Analysis
Prior work shows that temporal movement features can improve alert utility over frame-only triggers [4], [5], [16]. This motivates our rolling-window behavior logic.

### 2.4 Unified Surveillance Pipelines
Integrated detect-track-classify pipelines are commonly reported as more operationally useful than isolated detection modules [6], [7], [10], [11]. Our implementation follows this unified design with emphasis on CPU deployability.

## 3. System Methodology
### 3.1 Pipeline Overview
Each frame passes through five stages:
1. Video capture and preprocessing.
2. Human detection.
3. Animal foreground detection.
4. Temporal behavior classification.
5. Alert generation and monitoring output.

### 3.2 Human Detection Branch
Human candidates are detected via OpenCV HOG descriptor with linear SVM and non-maximum suppression. Candidate trajectories are tracked with centroid association.

### 3.3 Animal Detection Branch
MOG2 foreground masks are denoised with morphological filtering. Contours above minimum area thresholds are retained as animal candidates and tracked over time.

### 3.4 Temporal Classification and Alerts
Behavior labels are assigned from short-term movement signals (speed and dwell patterns). Alerts are generated for suspicious/lingering human behavior and prolonged animal-stationary states with cooldown logic to prevent duplicate alerts.

### 3.5 Key Implementation Parameters
Parameters used in evaluation runner include:
- Frame size: `640x480`
- HOG scale: `1.05`, stride: `(8, 8)`, padding: `(4, 4)`
- MOG2 history: `500`, variance threshold: `25`
- Minimum contour area: `800`
- Idle speed threshold: `2.0`, walking threshold: `10.0`
- Lingering threshold: `90` frames
- Animal stationary threshold: `1800` frames

## 4. Experimental Setup
### 4.1 Hardware and Software
- Device: MacBook Air (Apple M4, 10-core CPU; arm64)
- RAM: 16 GB
- OS: macOS 26.3.1 (Build 25D771280a)
- Software: Python 3.14.3, OpenCV 4.13.0

### 4.2 Data Sources
Primary labeled evaluation set:
- `horseVId.mp4` (56.97 s), mostly no-human baseline
- `horsevid2.mp4` (216.99 s), includes veterinary human interaction segments
- 28 non-overlapping reviewed clips (3 s each): 10 from `horseVId.mp4`, 18 from `horsevid2.mp4`

Additional runtime/qualitative examples:
- `human.mp4` (5.88 s)
- `animal.mp4` (5.88 s)

### 4.3 Metrics
- Human presence: precision, recall, F1, accuracy (clip-level)
- Event metrics: suspicious and animal-abnormal precision/recall when positives exist
- Behavior: accuracy on human-labeled clips
- Runtime: FPS and frame-time statistics
- Alert quality: false alarms/hour

## 5. Results
### 5.1 Primary Quantitative Results (Labeled Horse Set)
| Metric | Value |
|---|---:|
| Human precision (default threshold) | 28.6% |
| Human recall (default threshold) | 20.0% |
| Human F1 (default threshold) | 23.5% |
| Human precision (tuned threshold) | 52.6% |
| Human recall (tuned threshold) | 100.0% |
| Human F1 (tuned threshold) | 69.0% |
| Behavior accuracy (human-labeled clips) | 20.0% (2/10) |
| False alarms/hour (overall) | 52.56/hr |
| Throughput (weighted) | 51.89 FPS |

Additional task metrics on this set:
- Suspicious-event recall is not meaningful because ground-truth suspicious positives were 0 clips.
- Animal-abnormal detection recall was 0.0 on 2 positive clips.

### 5.2 Per-Source Breakdown (Labeled Horse Set)
| Source | Clips | Human Precision | Human Recall | Human F1 | Behavior Accuracy | FAR/hr | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| `horseVId.mp4` (no human-positive clips) | 10 | N/A | N/A | N/A | N/A | 0.00 | 55.54 |
| `horsevid2.mp4` (human-interaction segments) | 18 | 0.286 | 0.200 | 0.235 | 0.200 (2/10) | 66.36 | 50.93 |
| Combined (default threshold) | 28 | 0.286 | 0.200 | 0.235 | 0.200 (2/10) | 52.56 | 51.89 |
| Combined (tuned threshold) | 28 | 0.526 | 1.000 | 0.690 | 0.200 (2/10) | 52.56 | 51.89 |

### 5.3 Additional Validation and Runtime Examples (New Videos)
These videos are reported as qualitative/runtime checks only due short total duration and no manual labels.

| Video | Frames | Duration (s) | Processing FPS | Alerts | Human-detected Frames |
|---|---:|---:|---:|---:|---:|
| `human.mp4` | 141 | 5.88 | 44.55 | 0 | 141 (100.0%) |
| `animal.mp4` | 141 | 5.88 | 44.81 | 0 | 123 (87.2%) |
| Combined | 282 | 11.75 | 44.68 (weighted) | 0 | 264 (93.6%) |

### 5.4 Success Criteria Snapshot
| Criterion | Target | Observed | Status |
|---|---:|---:|---|
| Human precision | >= 0.80 | 0.286 (default), 0.526 (tuned) | Not Met |
| Behavior accuracy | >= 0.75 | 0.200 | Not Met |
| False alarms/hour | < 10 | 52.56 | Not Met |
| Runtime FPS | >= 15 | 51.89 | Met |

## 6. Discussion
### 6.1 Key Findings
1. The implementation is computationally efficient and sustains real-time throughput.
2. Human-detection thresholding drives a strong precision-recall tradeoff.
3. Current animal-abnormal logic lacks sensitivity for subtle handling events.
4. Dataset size and class balance dominate metric reliability.

### 6.2 Limitations
- Small evaluation set (28 clips) with class imbalance.
- Zero positive suspicious clips prevents meaningful suspicious recall reporting.
- Short clip windows can miss temporal context around transitions.
- Classical detectors are more brittle under occlusion/lighting changes than modern deep detectors.

### 6.3 Comparison to Related Work
Our runtime outcome aligns with the low-resource advantage of classical CV baselines [2], [3], while accuracy gaps indicate why recent literature increasingly uses deep detection/tracking models [7], [11], [18]. The results support this project as a practical baseline rather than an endpoint model.

## 7. Conclusion and Future Work
This project delivers a functional end-to-end surveillance prototype with strong runtime performance on standard CPU hardware and transparent evaluation artifacts. The main contribution is an integrated pipeline that unifies human-presence monitoring and animal behavior analytics. The current quantitative results expose clear areas for improvement in detection robustness and abnormal-event sensitivity.

Future work:
1. Add a modern detector baseline (e.g., YOLOv8) for robustness comparison.
2. Expand and rebalance labeled data with sufficient positive events.
3. Add event-level latency metrics and richer behavior labels.
4. Improve abnormal-event logic with contextual interaction features.

## 8. Reproducibility Artifacts
- Project repository: `https://github.com/KushTheDoosh/team10-computational-perception-final-project`
- Branch used for submission: `main`
Primary files in project folder:
- `00_reports/final_project_writeup_submission_ready.md`
- `03_evaluation/output_horse_review_final/metrics_clip_level.json`
- `03_evaluation/output_horse_review_final/review_sheet_for_labeling 1(in).csv`
- `03_evaluation/output_new_videos_2026-05-03/results_summary.json`

## 9. References
[1] USDA National Agricultural Statistics Service, *Agricultural Economics and Land Ownership Survey*, 2022.  
[2] N. Dalal and B. Triggs, “Histograms of Oriented Gradients for Human Detection,” *CVPR*, 2005.  
[3] Z. Zivkovic, “Improved Adaptive Gaussian Mixture Model for Background Subtraction,” *ICPR*, 2004.  
[4] M. Mokhtari and A. Khamis, “Detection of Abnormal Behavior in Surveillance Videos Using Machine Learning,” *Journal of Ambient Intelligence and Humanized Computing*, 2021.  
[5] S. Kumar, A. Singh, and D. P. Toshniwal, “Behavior Classification in Livestock Using Video-Based Motion Analysis,” *Computers and Electronics in Agriculture*, 2018.  
[6] Q. Zhang and J. Zhang, “Detection and Classification of Animal Behavior in Videos Using Deep Learning,” *Applied Sciences*, 2020.  
[7] Y. Li and X. Chen, “A Survey of Visual Surveillance Systems Based on Deep Learning,” *IEEE Access*, 2020.  
[8] Y. Zhao and M. Li, “Real-time Human Detection in Surveillance Videos Using Deep Learning,” *IEEE Transactions on Industrial Informatics*, 2020.  
[9] H. Chen, Y. Wang, and J. Liu, “Smart Farming: IoT-based Animal Health Monitoring System,” *IEEE Internet of Things Journal*, 2020.  
[10] R. Sinha and S. Mehta, “Integrating Human Detection and Animal Monitoring Systems for Enhanced Surveillance,” *International Journal of Computer Applications*, 2019.  
[11] J. Redmon et al., “You Only Look Once: Unified, Real-Time Object Detection,” *CVPR*, 2016.  
[12] P. Viola and M. Jones, “Robust Real-Time Face Detection,” *IJCV*, 2004.  
[13] C. Stauffer and W. E. L. Grimson, “Adaptive Background Mixture Models for Real-time Tracking,” *CVPR*, 1999.  
[14] O. Barnich and M. Van Droogenbroeck, “ViBe: A Universal Background Subtraction Algorithm,” *IEEE TIP*, 2011.  
[15] J. F. Henriques et al., “High-Speed Tracking with Kernelized Correlation Filters,” *IEEE TPAMI*, 2015.  
[16] S. Neethirajan, “Recent Advances in Wearable Sensors for Animal Health Management,” *Sensing and Bio-Sensing Research*, 2020.  
[17] L. Zhang et al., “Is Faster R-CNN Doing Well for Pedestrian Detection?” *ECCV*, 2016.  
[18] W. Liu et al., “SSD: Single Shot MultiBox Detector,” *ECCV*, 2016.  
[19] A. Bewley et al., “Simple Online and Realtime Tracking,” *ICIP*, 2016.  
[20] C. Wang et al., “A Deep Learning Approach for Detecting Abnormal Behaviors of Livestock Using Video Surveillance,” *Computers and Electronics in Agriculture*, 2019.

---

**Final editing note before submission:** verify reference formatting/style and regenerate figures (pipeline, sample detections, and metrics chart) to match course template.
