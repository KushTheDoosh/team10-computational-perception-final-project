# Horse Video Evaluation Metrics (For Final Writeup)

- Reviewed clips: 28 (3s each)
- Ground-truth human-present clips: 10
- Ground-truth no-human clips: 18
- Ground-truth suspicious-event clips: 0
- Ground-truth animal-abnormal clips: 2

## Runtime
- `horseVId.mp4`: 55.54 FPS
- `horsevid2.mp4`: 50.93 FPS
- Weighted average: 51.89 FPS

## Human Presence Detection (Clip Level)
| Rule | Precision | Recall | F1 | Accuracy | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Default (ratio >= 0.12) | 0.286 | 0.200 | 0.235 | 0.536 | 2 | 5 | 8 | 13 |
| Tuned (ratio >= 0.00) | 0.526 | 1.000 | 0.690 | 0.679 | 10 | 9 | 0 | 9 |

## Other Tasks (Clip Level)
| Task | Precision | Recall | F1 | Accuracy | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Suspicious event | N/A | N/A | N/A | 1.000 | 0 | 0 | 0 | 28 |
| Animal abnormal event | N/A | 0.000 | N/A | 0.929 | 0 | 0 | 2 | 26 |

## Behavior Label Accuracy
- Accuracy on human-labeled clips: 0.200 (2/10)

## Notes
- Suspicious-event recall is not meaningful here because there are no ground-truth positive suspicious clips.
- Results are from curated horse-stall clips and should be reported as scenario-specific evaluation.
