"""OpenCV image, video and webcam object detection."""
import argparse
from collections import Counter
import csv
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
METHOD_NAMES = {"hog": "HOG + linear SVM", "yolo": "YOLO-Fastest 1.1"}


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    box: tuple[int, int, int, int]  # x, y, width, height


class HOGDetector:
    def __init__(self, margin=0.0):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.margin = margin

    def __call__(self, frame):
        height, width = frame.shape[:2]
        if height < 128 or width < 64:
            return []
        boxes, scores = self.hog.detectMultiScale(
            frame, hitThreshold=self.margin, winStride=(8, 8),
            padding=(8, 8), scale=1.05, groupThreshold=2.0,
        )
        return [
            Detection("person", float(score), tuple(map(int, box)))
            for box, score in zip(boxes, scores)
        ]


class YOLODetector:
    def __init__(self, confidence=0.25):
        directory = ROOT / "weights"
        self.labels = (directory / "coco.names").read_text().splitlines()
        weights = directory / "yolo-fastest-1.1.weights"
        if not weights.is_file():
            raise FileNotFoundError(f"Missing pretrained weights at {weights}")
        self.model = cv2.dnn_DetectionModel(
            str(weights), str(directory / "yolo-fastest-1.1.cfg")
        )
        self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.model.setInputParams(size=(320, 320), scale=1 / 255, swapRB=True)
        self.model.setNmsAcrossClasses(False)
        self.confidence = confidence

    def __call__(self, frame):
        classes, scores, boxes = self.model.detect(
            frame, confThreshold=self.confidence, nmsThreshold=0.45
        )
        height, width = frame.shape[:2]
        detections = []
        for category, score, (x, y, w, h) in zip(classes, scores, boxes):
            left, top = max(0, int(x)), max(0, int(y))
            right, bottom = min(width, int(x + w)), min(height, int(y + h))
            if right > left and bottom > top:
                detections.append(Detection(
                    self.labels[int(category)], float(score),
                    (left, top, right - left, bottom - top),
                ))
        return detections


def make_detector(method, confidence=0.25):
    return YOLODetector(confidence) if method == "yolo" else HOGDetector()


def counts_for(detections):
    return Counter(item.label for item in detections)


def annotate(frame, detections, method, elapsed_ms, panel_lines=0):
    image = frame.copy()
    height, width = image.shape[:2]
    scale = max(0.38, min(0.72, width / 1000))
    box_color = (82, 218, 92) if method == "yolo" else (10, 186, 255)
    for item in sorted(detections, key=lambda detection: detection.score):
        x, y, w, h = item.box
        cv2.rectangle(image, (x, y), (x + w - 1, y + h - 1), box_color, 2)
        score = f"{item.score:.2f}" if method == "yolo" else f"margin {item.score:+.2f}"
        label = f"{item.label} {score}"
        (text_width, text_height), baseline = cv2.getTextSize(label, 0, scale, 1)
        text_x = max(0, min(x, width - text_width - 6))
        text_y = max(text_height + 5, y)
        cv2.rectangle(
            image, (text_x, text_y - text_height - 5),
            (text_x + text_width + 5, text_y + baseline), box_color, -1,
        )
        cv2.putText(image, label, (text_x + 2, text_y - 2), 0, scale,
                    (18, 24, 25), 1, cv2.LINE_AA)

    counts = counts_for(detections)
    summary = f"{METHOD_NAMES[method]} | {len(detections)} objects | {elapsed_ms:.1f} ms"
    lines = [summary]
    count_line = ""
    for label, count in sorted(counts.items()):
        item = f"{label}: {count}"
        candidate = f"{count_line}  |  {item}" if count_line else item
        if count_line and cv2.getTextSize(candidate, 0, scale, 1)[0][0] > width - 18:
            lines.append(count_line)
            count_line = item
        else:
            count_line = candidate
    lines.append(count_line or "No detections")
    step = max(17, round(30 * scale))
    panel_width = max(width, cv2.getTextSize(summary, 0, scale, 1)[0][0] + 18)
    canvas = np.full((height + step * max(len(lines), panel_lines) + 10, panel_width, 3), 24, np.uint8)
    canvas[:height, :width] = image
    for index, line in enumerate(lines):
        cv2.putText(canvas, line, (9, height + step * (index + 1)),
                    0, scale, (247, 249, 251), 1, cv2.LINE_AA)
    return canvas


def read_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image {path}")
    return image


def save_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write image {path}")


def timed_detection(detector, frame):
    start = perf_counter()
    detections = detector(frame)
    return detections, (perf_counter() - start) * 1000


def count_rows(frame_id, detections, elapsed):
    counts = counts_for(detections)
    items = sorted(counts.items()) or [("", 0)]
    for label, count in items:
        yield frame_id, label, count, round(elapsed, 3)


def process_media(args, detector):
    source = Path(args.source)
    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "counts.csv"
    if source.is_dir() or source.suffix.lower() in IMAGE_TYPES:
        images = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_TYPES) if source.is_dir() else [source]
        if not images:
            raise ValueError(f"No images found in {source}")
        with csv_path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["image", "class", "count", "processing_ms"])
            for path in images:
                frame = read_image(path)
                detections, duration = timed_detection(detector, frame)
                annotated = annotate(frame, detections, args.method, duration)
                save_image(args.output / f"{path.stem}_{path.suffix[1:]}_annotated.jpg", annotated)
                writer.writerows(count_rows(path.name, detections, duration))
        return

    if not args.source.isdecimal() and not source.is_file():
        raise ValueError(f"Expected an image, folder, video file or webcam index: {source}")
    camera = not source.is_file()
    capture = cv2.VideoCapture(int(args.source) if camera else str(source))
    writer_video = None
    frames = 0
    expected_size = None
    try:
        if not capture.isOpened():
            raise OSError(f"Could not open {args.source}. Check the path or webcam permissions.")
        fps = capture.get(cv2.CAP_PROP_FPS)
        fps = fps if 0 < fps <= 240 else 24.0
        with csv_path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["frame", "class", "count", "processing_ms"])
            while args.max_frames is None or frames < args.max_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                frames += 1
                detections, duration = timed_detection(detector, frame)
                output = annotate(frame, detections, args.method, duration, panel_lines=3)
                if writer_video is None and args.save_video:
                    expected_size = (output.shape[1] // 2 * 2, output.shape[0] // 2 * 2)
                    writer_video = cv2.VideoWriter(
                        str(args.output / "annotated.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"), fps, expected_size,
                    )
                    if not writer_video.isOpened():
                        raise OSError("Could not create annotated.mp4")
                if writer_video is not None:
                    if (output.shape[1], output.shape[0]) != expected_size:
                        video_frame = cv2.resize(output, expected_size)
                    else:
                        video_frame = output
                    writer_video.write(video_frame)
                writer.writerows(count_rows(frames, detections, duration))
                if args.show:
                    cv2.imshow("Object detection and counting", output)
                    wait = 1 if camera else max(1, round(1000 / fps - duration))
                    if cv2.waitKey(wait) & 0xFF in (ord("q"), 27):
                        break
        if frames == 0:
            raise OSError(f"No frames were decoded from {args.source}")
        print(f"Processed {frames} frames")
    finally:
        capture.release()
        if writer_video is not None:
            writer_video.release()
        if args.show:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Image, image folder, video file, or webcam index")
    parser.add_argument("--method", choices=METHOD_NAMES, default="yolo")
    parser.add_argument("--confidence", type=float, default=0.25, help="YOLO threshold")
    parser.add_argument("--output", type=Path, default=Path("run"))
    parser.add_argument("--show", action="store_true", help="Open a video or webcam window")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    if not 0 < args.confidence <= 1 or (args.max_frames is not None and args.max_frames < 1):
        parser.error("Confidence must be in (0, 1] and max-frames must be positive.")
    if args.source.isdecimal() and not (args.show or args.max_frames):
        parser.error("For a webcam, use --show or --max-frames.")
    cv2.setNumThreads(1)
    try:
        process_media(args, make_detector(args.method, args.confidence))
    except (OSError, ValueError, cv2.error) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Outputs: {args.output.resolve()}")


if __name__ == "__main__":
    main()
