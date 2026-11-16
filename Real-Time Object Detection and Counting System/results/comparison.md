# Measured detector comparison

Python 3.13.9, OpenCV 4.13.0, Darwin arm64, CPU, one OpenCV thread.

Person boxes were hand-marked before scoring. Greedy one-to-one matching uses IoU ≥ 0.50. Timing covers preprocessing, detection and postprocessing. Each image uses the median of five runs after three warm-up calls. The latency column is the median across the eight image medians.

| Detector | TP | FP | FN | Precision | Recall | F1 | Person count MAE | Median ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HOG + linear SVM | 2 | 9 | 24 | 0.1818 | 0.0769 | 0.1081 | 2.125 | 149.635 |
| YOLO-Fastest 1.1 | 17 | 7 | 9 | 0.7083 | 0.6538 | 0.68 | 1 | 12.762 |

## Per-image person counts

| Image | Reference | HOG | YOLO | HOG ms | YOLO ms |
|---|---:|---:|---:|---:|---:|
| 01_bus.jpg | 4 | 3 | 2 | 355.326 | 12.791 |
| 02_team.jpg | 2 | 1 | 2 | 373.592 | 12.878 |
| 03_court.png | 2 | 2 | 2 | 110.02 | 12.939 |
| 04_cyclist.jpg | 2 | 0 | 2 | 51.028 | 12.631 |
| 05_street.jpg | 0 | 1 | 1 | 164.777 | 12.753 |
| 06_bus_low_light.png | 4 | 1 | 3 | 351.14 | 12.771 |
| 07_court_small.png | 2 | 1 | 0 | 134.493 | 12.729 |
| 08_crowded_montage.png | 10 | 2 | 12 | 116.905 | 12.734 |

## Predicted category totals

- HOG + linear SVM — person: 11
- YOLO-Fastest 1.1 — bicycle: 2, bus: 2, car: 2, cat: 1, dog: 1, person: 24, remote: 1, tennis racket: 2, truck: 1

Totals sum detections across the eight test cases. The derived cases reuse source objects. Person metrics use the included annotations. Other category counts are model predictions.
