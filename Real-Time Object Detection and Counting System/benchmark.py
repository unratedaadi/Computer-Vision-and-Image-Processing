"""Measure both methods on the included ground-truth person boxes."""
from collections import Counter
import csv
import json
import platform
from statistics import mean, median

import cv2

from detect import (
    METHOD_NAMES, ROOT, annotate, counts_for, make_detector, read_image,
    save_image, timed_detection,
)


def intersection_over_union(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    w = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    h = max(0, min(ay + ah, by + bh) - max(ay, by))
    intersection = w * h
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0


def score_people(detections, ground_truth):
    unmatched = dict(enumerate(ground_truth))
    true_positives = false_positives = 0
    people = sorted((item for item in detections if item.label == "person"),
                    key=lambda item: item.score, reverse=True)
    for item in people:
        overlaps = {i: intersection_over_union(item.box, box)
                    for i, box in unmatched.items()}
        best = max(overlaps, key=overlaps.get, default=None)
        if best is not None and overlaps[best] >= 0.5:
            true_positives += 1
            del unmatched[best]
        else:
            false_positives += 1
    return true_positives, false_positives, len(unmatched)


def write_csv(path, fields, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    cv2.setNumThreads(1)
    samples = ROOT / "inputs"
    results = ROOT / "results"
    cases = json.loads((samples / "person_ground_truth.json").read_text())
    scores, counts, boxes, summary, groups = [], [], [], [], []
    totals_by_method = {}
    images = {name: read_image(samples / name) for name in cases}
    for method in METHOD_NAMES:
        detector = make_detector(method)
        destination = results / method
        destination.mkdir(parents=True, exist_ok=True)
        for _ in range(3):
            detector(next(iter(images.values())))
        totals = Counter()
        class_totals = Counter()
        times, errors = [], []
        for name, case in cases.items():
            frame = images[name]
            truth = case["people"]
            samples_ms = []
            for _ in range(5):
                detections, elapsed = timed_detection(detector, frame)
                samples_ms.append(elapsed)
            duration = median(samples_ms)
            times.append(duration)
            save_image(destination / f"{name}.annotated.jpg",
                       annotate(frame, detections, method, duration))
            class_counts = counts_for(detections)
            class_totals.update(class_counts)
            for label, count in sorted(class_counts.items()) or [("", 0)]:
                counts.append(dict(method=method, image=name, label=label, count=count))
            for item in detections:
                x, y, w, h = item.box
                boxes.append(dict(method=method, image=name, label=item.label,
                                  score=round(item.score, 5), x=x, y=y, width=w, height=h))
            tp, fp, fn = score_people(detections, truth)
            totals.update(tp=tp, fp=fp, fn=fn)
            error = abs(class_counts["person"] - len(truth))
            errors.append(error)
            scores.append(dict(method=method, image=name, condition=case["condition"],
                               people_actual=len(truth), people_detected=class_counts["person"],
                               tp=tp, fp=fp, fn=fn, absolute_count_error=error,
                               processing_ms=round(duration, 3)))
        tp, fp, fn = (totals[key] for key in ("tp", "fp", "fn"))
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        summary.append(dict(method=method, tp=tp, fp=fp, fn=fn,
                            precision=round(precision, 4), recall=round(recall, 4),
                            f1=round(2 * tp / (2 * tp + fp + fn), 4) if tp + fp + fn else 0,
                            person_count_mae=round(mean(errors), 3),
                            median_ms=round(median(times), 3)))
        totals_by_method[method] = class_totals
        groups.extend(dict(method=method, label=label, count=count)
                      for label, count in sorted(class_totals.items()))

    write_csv(results / "object_counts.csv", ["method", "image", "label", "count"], counts)
    write_csv(results / "category_totals.csv", ["method", "label", "count"], groups)
    write_csv(results / "detections.csv",
              ["method", "image", "label", "score", "x", "y", "width", "height"], boxes)
    write_csv(results / "per_image.csv", list(scores[0]), scores)
    write_csv(results / "comparison.csv", list(summary[0]), summary)

    lines = [
        "# Measured detector comparison", "",
        f"Python {platform.python_version()}, OpenCV {cv2.__version__}, "
        f"{platform.system()} {platform.machine()}, CPU, one OpenCV thread.", "",
        "Person boxes were hand-marked before scoring. Greedy one-to-one matching "
        "uses IoU ≥ 0.50. Timing covers preprocessing, detection and postprocessing. "
        "Each image uses the median of five runs after three warm-up calls. "
        "The latency column is the median across the eight image medians.", "",
        "| Detector | TP | FP | FN | Precision | Recall | F1 | Person count MAE | Median ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        cells = [METHOD_NAMES[row["method"]], *list(row.values())[1:]]
        lines.append("| " + " | ".join(map(str, cells)) + " |")
    lines.extend(["", "## Per-image person counts", "",
                  "| Image | Reference | HOG | YOLO | HOG ms | YOLO ms |",
                  "|---|---:|---:|---:|---:|---:|"])
    for hog, yolo in zip(scores[:len(cases)], scores[len(cases):]):
        lines.append(f'| {hog["image"]} | {hog["people_actual"]} | {hog["people_detected"]} | '
                     f'{yolo["people_detected"]} | {hog["processing_ms"]} | {yolo["processing_ms"]} |')
    lines.extend(["", "## Predicted category totals", ""])
    for method, totals in totals_by_method.items():
        lines.append(f'- {METHOD_NAMES[method]} — ' +
                     ", ".join(f"{label}: {count}" for label, count in sorted(totals.items())))
    lines.extend(["", "Totals sum detections across the eight test cases. "
                  "The derived cases reuse source objects. Person metrics use the "
                  "included annotations. Other category counts are model predictions.", ""])
    (results / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(results / "comparison.md")


if __name__ == "__main__":
    main()
