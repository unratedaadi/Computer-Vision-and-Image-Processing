"""Evaluate three OpenCV image enhancement methods on one image or a folder."""

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

ROOT = Path(__file__).resolve().parent
EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
METHODS = {
    "original": "Original input",
    "brightness": "Brightness adjustment",
    "stretch": "Contrast stretching",
    "equalization": "Histogram equalization",
}
FIELDS = (
    "image", "method", "input_diagnosis", "mean_luma", "std_luma",
    "p05_p95_span", "dark_pct", "white_pct", "psnr_db", "ssim",
)


def read_image(path):
    # imdecode also supports Unicode file paths on Windows.
    data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def write_png(path, image):
    encoded, data = cv2.imencode(".png", image)
    if not encoded:
        raise ValueError(f"Cannot encode image: {path}")
    path.write_bytes(data.tobytes())


def luma(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[:, :, 0]


def with_luma(ycrcb, channel):
    result = ycrcb.copy()
    result[:, :, 0] = channel
    return cv2.cvtColor(result, cv2.COLOR_YCrCb2BGR)


def stretch(channel):
    low, high = np.percentile(channel, [2, 98])
    if high <= low:
        return channel.copy()
    return np.rint(np.clip((channel.astype(np.float32) - low)
                          * (255 / (high - low)), 0, 255)).astype(np.uint8)


def enhance(image):
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    channel = ycrcb[:, :, 0]
    return {
        "original": image,
        "brightness": with_luma(ycrcb, cv2.add(channel, 45)),
        "stretch": with_luma(ycrcb, stretch(channel)),
        "equalization": with_luma(ycrcb, cv2.equalizeHist(channel)),
    }


def luma_metrics(channel):
    low, high = np.percentile(channel, [5, 95])
    return {
        "mean_luma": float(channel.mean()),
        "std_luma": float(channel.std()),
        "p05_p95_span": float(high - low),
        "dark_pct": float(100 * np.mean(channel <= 15)),
        "white_pct": float(100 * np.mean(channel >= 250)),
    }


def diagnose(metrics):
    issues = []
    if metrics["mean_luma"] < 90:
        issues.append("low brightness")
    if metrics["p05_p95_span"] < 110:
        issues.append("low contrast")
    return " and ".join(issues) or "neither flagged"


def reference_metrics(image, reference):
    if reference is None:
        return {"psnr_db": None, "ssim": None}
    psnr = (float("inf") if np.array_equal(image, reference)
            else peak_signal_noise_ratio(reference, image, data_range=255))
    min_side = min(image.shape[:2])
    window = min(7, min_side if min_side % 2 else min_side - 1)
    ssim = None
    if window >= 3:
        ssim = structural_similarity(reference, image, channel_axis=2,
                                     data_range=255, win_size=window)
    return {"psnr_db": psnr, "ssim": ssim}


def comparison_figure(path, title, images, channels, reference):
    panels = {METHODS[key]: image for key, image in images.items()}
    if reference is not None:
        panels["Clean reference"] = reference

    fig = plt.figure(figsize=(13, 6), facecolor="#f5f7fb")
    grid = fig.add_gridspec(2, len(panels), height_ratios=[1.4, 1],
                           left=0.085, right=0.98, top=0.85, bottom=0.11,
                           hspace=0.36, wspace=0.09)
    fig.suptitle(title, y=0.95, fontsize=19, color="#17243b", fontweight="bold")
    for index, (label, image) in enumerate(panels.items()):
        ax = fig.add_subplot(grid[0, index])
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title(label, fontsize=11)
        ax.axis("off")

    ax = fig.add_subplot(grid[1, :])
    colors = ("#404b5b", "#007c78", "#ba5318", "#433da0")
    for (method, channel), color in zip(channels.items(), colors):
        hist = np.bincount(channel.ravel(), minlength=256) * (100 / channel.size)
        ax.plot(hist, color=color, label=METHODS[method], linewidth=1.25)
    if reference is not None:
        channel = luma(reference)
        hist = np.bincount(channel.ravel(), minlength=256) * (100 / channel.size)
        ax.plot(hist, label="Clean reference", color="#1d648f",
                linestyle="--", linewidth=1.1)
    ax.set(xlim=(0, 255), xlabel="Luma (0–255)", ylabel="Pixels per luma level (%)")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(ncol=len(panels), fontsize=9, loc="lower center",
              bbox_to_anchor=(0.5, 1.015), frameon=False)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def image_paths(path):
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(item for item in path.iterdir()
                      if item.is_file() and item.suffix.lower() in EXTENSIONS)
    raise ValueError(f"Path does not exist: {path}")


def process(source, output, reference_dir, show):
    paths = image_paths(source)
    if not paths:
        raise ValueError(f"No supported images found in {source}")
    if len({path.stem.casefold() for path in paths}) != len(paths):
        raise ValueError("Input filenames must have distinct stems.")
    references = {}
    if reference_dir is not None:
        if not reference_dir.is_dir():
            raise ValueError(f"Reference path must be a directory: {reference_dir}")
        for path in image_paths(reference_dir):
            if path.stem in references:
                raise ValueError(f"Duplicate reference stem: {path.stem}")
            references[path.stem] = path

    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in paths:
        original = read_image(path)
        reference_path = references.get(path.stem)
        reference = read_image(reference_path) if reference_path else None
        if reference is not None and reference.shape != original.shape:
            raise ValueError(f"Reference dimensions differ for {path.name}")
        images = enhance(original)
        channels = {method: luma(image) for method, image in images.items()}
        diagnosis = diagnose(luma_metrics(channels["original"]))
        image_output = output / path.stem
        image_output.mkdir(exist_ok=True)

        for method, image in images.items():
            if method != "original":
                write_png(image_output / f"{method}.png", image)
            metrics = {**luma_metrics(channels[method]),
                       **reference_metrics(image, reference)}
            rows.append({
                "image": path.name,
                "method": method,
                "input_diagnosis": diagnosis,
                **{key: "" if value is None else f"{value:.6f}"
                   for key, value in metrics.items()},
            })

        figure = image_output / "comparison.png"
        title = f"{path.stem.replace('_', ' ').title()}  |  {diagnosis.capitalize()}"
        comparison_figure(figure, title, images, channels, reference)
        print(f"{path.name}: {diagnosis}")
        if show:
            preview = read_image(figure)
            scale = min(1, 1200 / preview.shape[1], 700 / preview.shape[0])
            if scale < 1:
                preview = cv2.resize(preview, None, fx=scale, fy=scale,
                                     interpolation=cv2.INTER_AREA)
            try:
                window = "Image enhancement | any key to continue"
                cv2.imshow(window, preview)
                cv2.waitKey(0)
            finally:
                cv2.destroyAllWindows()

    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(paths)} comparisons and {len(rows)} metric rows to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=ROOT / "images/input",
                        help="Image file or directory (default: bundled benchmark)")
    parser.add_argument("--output", type=Path, default=ROOT / "results",
                        help="Directory for PNG comparisons and metrics.csv")
    parser.add_argument("--reference-dir", type=Path,
                        help="Clean image folder, matched by stem and dimensions")
    parser.add_argument("--show", action="store_true",
                        help="Open previews using OpenCV on a desktop")
    args = parser.parse_args()
    references = args.reference_dir
    if references is None and args.input.resolve() == ROOT / "images/input":
        references = ROOT / "images/reference"
    try:
        process(args.input, args.output, references, args.show)
    except (OSError, ValueError, cv2.error) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
