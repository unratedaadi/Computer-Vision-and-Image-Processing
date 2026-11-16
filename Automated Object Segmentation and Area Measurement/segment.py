"""Segment objects, measure foreground pixels and compare thresholding methods."""
import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
METHODS = ("global", "otsu")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class Settings:
    threshold: int = 200
    polarity: str = "dark"
    kernel: int = 5
    min_area: int = 100


def load_image(path, mode=cv2.IMREAD_COLOR):
    # Decode file bytes to support paths with non-ASCII characters.
    encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    if encoded.size == 0:
        raise ValueError(f"Empty image: {path}")
    image = cv2.imdecode(encoded, mode)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def save_image(path, image):
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise OSError(f"Cannot encode image: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded.tobytes())


def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def segment(gray, method, settings):
    mode = cv2.THRESH_BINARY_INV if settings.polarity == "dark" else cv2.THRESH_BINARY
    if method == "otsu":
        threshold, raw = cv2.threshold(gray, 0, 255, mode | cv2.THRESH_OTSU)
    elif method == "global":
        threshold, raw = cv2.threshold(gray, settings.threshold, 255, mode)
    else:
        raise ValueError(f"Unknown method: {method}")

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (settings.kernel, settings.kernel))
    cleaned = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    _, labels, stats, centers = cv2.connectedComponentsWithStats(cleaned, connectivity=8)

    kept = [label for label in range(1, len(stats)) if stats[label, cv2.CC_STAT_AREA] >= settings.min_area]
    lookup = np.zeros(len(stats), dtype=np.uint8)
    lookup[kept] = 255

    objects = []
    for object_id, label in enumerate(kept, start=1):
        x, y, width, height, area = map(int, stats[label])
        center_x, center_y = centers[label]
        objects.append({
            "object_id": object_id,
            "area_px": area,
            "x": x,
            "y": y,
            "width_px": width,
            "height_px": height,
            "centroid_x": round(float(center_x), 2),
            "centroid_y": round(float(center_y), 2),
        })
    return int(threshold), raw, lookup[labels], objects


def annotate(image, mask, objects):
    overlay = image.copy()
    pixels = mask != 0
    overlay[pixels] = (100, 215, 45)
    overlay = cv2.addWeighted(image, 0.73, overlay, 0.27, 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (70, 235, 100), 2)
    for obj in objects:
        label = f'{obj["object_id"]}: {obj["area_px"]:,} px^2'
        (text_width, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        x = max(0, min(int(obj["centroid_x"]) - text_width // 2, image.shape[1] - text_width - 10))
        y = max(18, min(int(obj["centroid_y"]), image.shape[0] - 5))
        cv2.rectangle(overlay, (x, y - 15), (x + text_width + 8, y + 4), (20, 38, 22), -1)
        cv2.putText(overlay, label, (x + 4, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return overlay


def mask_scores(prediction, reference):
    predicted = prediction != 0
    actual = reference != 0
    intersection = np.count_nonzero(predicted & actual)
    union = np.count_nonzero(predicted | actual)
    total = np.count_nonzero(predicted) + np.count_nonzero(actual)
    return {
        "iou": intersection / union if union else 1.0,
        "dice": 2 * intersection / total if total else 1.0,
        "pixel_accuracy": float(np.mean(predicted == actual)),
    }


def compare_view(image, reference, results):
    height, width = image.shape[:2]
    scale = min(1.0, 380 / width, 245 / height)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    panels = [("Input image", image)]
    if reference is not None:
        panels.append(("Reference foreground", reference))
    else:
        panels.append(("Grayscale after Gaussian blur", preprocess(image)))

    for method in METHODS:
        overlay, mask, count, threshold = results[method]
        title = "Fixed global" if method == "global" else "Otsu"
        panels.extend([
            (f"{title} mask | T = {threshold}", mask),
            (f"{title} output | {count} detected", overlay),
        ])

    tiles = []
    for title, panel in panels:
        nearest = title.endswith("foreground") or " mask " in title
        interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
        if panel.ndim == 2:
            panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
        panel = cv2.resize(panel, size, interpolation=interpolation)
        tile = np.full((278, 400, 3), (38, 33, 23), dtype=np.uint8)
        x = (400 - size[0]) // 2
        y = 27 + (245 - size[1]) // 2
        tile[y:y + size[1], x:x + size[0]] = panel
        cv2.putText(tile, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (244, 246, 249), 1, cv2.LINE_AA)
        tiles.append(tile)
    return np.vstack([np.hstack(tiles[index:index + 2]) for index in range(0, 6, 2)])


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


SUMMARY_FIELDS = [
    "image", "condition", "method", "threshold", "width_px", "height_px", "objects",
    "area_px", "reference_objects", "reference_area_px", "absolute_count_error",
    "area_error_pct", "raw_iou", "iou", "dice", "pixel_accuracy",
]
OBJECT_FIELDS = [
    "image", "method", "object_id", "area_px", "x", "y", "width_px", "height_px",
    "centroid_x", "centroid_y",
]


def run(paths, output, settings, cases=None, reference_dir=None):
    cases = cases or {}
    output.mkdir(parents=True, exist_ok=True)
    summary, measurements, previews = [], [], []

    for path in paths:
        name = path.stem
        image = load_image(path)
        gray = preprocess(image)
        height, width = gray.shape
        case = cases.get(name, {})
        reference_path = reference_dir / f"{name}.png" if reference_dir else None
        reference = None
        if reference_path and reference_path.is_file():
            reference = load_image(reference_path, cv2.IMREAD_GRAYSCALE)
            if reference.shape != gray.shape or np.any((reference != 0) & (reference != 255)):
                raise ValueError(f"Reference must be a 0/255 mask at input dimensions: {reference_path}")

        image_output = output / name
        save_image(image_output / "grayscale.png", cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        results = {}
        for method in METHODS:
            threshold, raw, mask, objects = segment(gray, method, settings)
            overlay = annotate(image, mask, objects)
            save_image(image_output / method / "raw_mask.png", raw)
            save_image(image_output / method / "mask.png", mask)
            save_image(image_output / method / "annotated.png", overlay)
            area = sum(obj["area_px"] for obj in objects)
            row = {
                "image": name, "condition": case.get("condition", "custom"), "method": method,
                "threshold": threshold, "width_px": width, "height_px": height,
                "objects": len(objects), "area_px": area,
            }
            if "count" in case:
                count = case["count"]
                row.update(reference_objects=count, absolute_count_error=abs(len(objects) - count))
            if reference is not None:
                ref_area = int(np.count_nonzero(reference))
                row["reference_area_px"] = ref_area
                if ref_area:
                    row["area_error_pct"] = round(100 * (area - ref_area) / ref_area, 3)
                row["raw_iou"] = round(mask_scores(raw, reference)["iou"], 6)
                row.update({key: round(value, 6) for key, value in mask_scores(mask, reference).items()})
            summary.append(row)
            measurements.extend({"image": name, "method": method, **obj} for obj in objects)
            results[method] = (overlay, mask, len(objects), threshold)

        preview = image_output / "comparison.png"
        save_image(preview, compare_view(image, reference, results))
        previews.append(preview)

    write_csv(output / "summary.csv", SUMMARY_FIELDS, summary)
    write_csv(output / "objects.csv", OBJECT_FIELDS, measurements)
    return summary, previews


def show_previews(previews):
    if not cv2.currentUIFramework():
        raise ValueError("OpenCV GUI support is required for --show. Open comparison.png files instead.")
    title = "Segmentation | N next, P previous, Q quit"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 800, 830)
    index = 0
    try:
        while True:
            cv2.imshow(title, load_image(previews[index]))
            key = cv2.waitKey(50) & 0xFF
            if key in (27, ord("q")) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("p"):
                index = (index - 1) % len(previews)
            if key in (ord("n"), 32):
                index = (index + 1) % len(previews)
    finally:
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Image or directory. Omit for bundled dataset.")
    parser.add_argument("--output", type=Path, help="Output directory. Defaults to outputs or custom-output.")
    parser.add_argument("--reference", type=Path, help="Directory of same-stem binary PNG reference masks.")
    parser.add_argument("--threshold", type=int, default=200)
    parser.add_argument("--polarity", choices=("dark", "light"), default="dark")
    parser.add_argument("--kernel", type=int, default=5)
    parser.add_argument("--min-area", type=int, default=100, help="Minimum component area in image pixels.")
    parser.add_argument("--show", action="store_true", help="Show side-by-side results in an OpenCV window.")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 255 or args.kernel < 1 or args.kernel % 2 == 0 or args.min_area < 1:
        parser.error("Threshold must be 0–255, kernel must be odd and positive, and min-area must be positive.")

    bundled = args.input is None
    source = args.input or DATA / "inputs"
    paths = sorted(path for path in source.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES) \
        if source.is_dir() else [source] if source.is_file() else []
    if not paths:
        parser.error(f"No supported images found at {source}.")
    if len({path.stem for path in paths}) != len(paths):
        parser.error("Each input image must have a unique filename stem.")
    if args.reference and not args.reference.is_dir():
        parser.error(f"Reference mask directory does not exist: {args.reference}.")

    settings = Settings(args.threshold, args.polarity, args.kernel, args.min_area)
    cases = json.loads((DATA / "cases.json").read_text(encoding="utf-8")) if bundled else {}
    if bundled:
        case_order = {name: position for position, name in enumerate(cases)}
        paths.sort(key=lambda path: case_order.get(path.stem, len(cases)))
    reference = args.reference or (DATA / "reference" if bundled else None)
    output = args.output or (ROOT / "outputs" if bundled else ROOT / "custom-output")
    try:
        summary, previews = run(paths, output, settings, cases, reference)
        for row in summary:
            print(f'{row["image"]:<24} {row["method"]:<6}  '
                  f'T={row["threshold"]:>3}  count={row["objects"]:>2}  area={row["area_px"]:>7,} px^2')
        print(f"\nTables and comparison images saved to {output.resolve()}")
        if args.show:
            show_previews(previews)
    except (OSError, ValueError, cv2.error) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
