"""Compare three spatial filters on two reproducible types of image noise."""

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
METHODS = ("noisy", "mean", "gaussian", "median")


def add_impulse_noise(image, amount, rng):
    noisy = image.copy()
    height, width = image.shape[:2]
    indexes = rng.choice(height * width, int(round(amount * height * width)), replace=False)
    salt, pepper = np.array_split(indexes, 2)
    noisy[salt // width, salt % width] = 255
    noisy[pepper // width, pepper % width] = 0
    return noisy


def add_gaussian_noise(image, sigma, rng):
    noise = rng.normal(0, sigma, size=image.shape)
    return np.clip(np.rint(image.astype(np.float64) + noise), 0, 255).astype(np.uint8)


def filter_image(noisy):
    return {
        "noisy": noisy,
        "mean": cv2.blur(noisy, (3, 3), borderType=cv2.BORDER_REPLICATE),
        "gaussian": cv2.GaussianBlur(noisy, (3, 3), 0.85, borderType=cv2.BORDER_REPLICATE),
        "median": cv2.medianBlur(noisy, 3),
    }


def quality(reference, candidate):
    error = reference.astype(np.float64) - candidate.astype(np.float64)
    mse = float(np.mean(np.square(error)))
    psnr = math.inf if mse == 0 else 10 * math.log10(255**2 / mse)
    return mse, psnr


def comparison_panel(tiles):
    tile_width, tile_height, header = 288, 206, 51
    canvas = np.full((2 * (tile_height + header), 3 * tile_width, 3), 30, np.uint8)
    for i, (label, detail, image) in enumerate(tiles):
        row, column = divmod(i, 3)
        x, y = column * tile_width, row * (tile_height + header)
        cv2.putText(canvas, label, (x + 10, y + 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, detail, (x + 10, y + 41), cv2.FONT_HERSHEY_SIMPLEX,
                    0.47, (192, 213, 231), 1, cv2.LINE_AA)
        if image is None:
            continue
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        height, width = image.shape[:2]
        scale = min((tile_width - 16) / width, (tile_height - 12) / height, 1)
        display = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))),
                             interpolation=cv2.INTER_AREA)
        h, w = display.shape[:2]
        left, top = x + (tile_width - w) // 2, y + header + (tile_height - h) // 2
        canvas[top:top + h, left:left + w] = display
    return canvas


def write_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write {path}")


def write_csv(path, records):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def run(paths, output, seed, impulse_amount, gaussian_sigma):
    rng = np.random.default_rng(seed)
    rows = []
    panels = []
    for position, path in enumerate(paths, start=1):
        source = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if source is None or source.dtype != np.uint8:
            raise ValueError(f"Expected a readable 8-bit image: {path}")
        if source.ndim == 3 and source.shape[2] == 4:
            source = cv2.cvtColor(source, cv2.COLOR_BGRA2BGR)
        image_dir = output / f"{position:02d}_{path.stem}"
        image_dir.mkdir(parents=True, exist_ok=True)
        write_image(image_dir / "original.png", source)
        conditions = (
            ("salt_pepper", f"Salt-and-pepper | {impulse_amount:.1%} pixels",
             add_impulse_noise(source, impulse_amount, rng)),
            ("gaussian", f"Gaussian noise | sigma {gaussian_sigma:g}",
             add_gaussian_noise(source, gaussian_sigma, rng)),
        )
        for noise, title, noisy in conditions:
            tiles = [("Original", "Clean reference", source)]
            for method, result in filter_image(noisy).items():
                mse, psnr = quality(source, result)
                rows.append(dict(image=path.name, noise=noise, method=method, mse=mse, psnr_db=psnr))
                tiles.append((method.capitalize() if method == "noisy" else f"{method.capitalize()} filter",
                              f"PSNR {psnr:.2f} dB | MSE {mse:.2f}", result))
                write_image(image_dir / f"{noise}_{method}.png", result)
            tiles.append((path.stem.title(), title, None))
            panel_path = image_dir / f"{noise}_comparison.png"
            write_image(panel_path, comparison_panel(tiles))
            panels.append(panel_path)
    write_csv(output / "per_image_metrics.csv", rows)
    summary = []
    for noise in ("salt_pepper", "gaussian"):
        for method in METHODS:
            sample = [row for row in rows if row["noise"] == noise and row["method"] == method]
            summary.append(dict(noise=noise, method=method, images=len(sample),
                                mean_mse=math.fsum(row["mse"] for row in sample) / len(sample),
                                mean_psnr_db=math.fsum(row["psnr_db"] for row in sample) / len(sample)))
    write_csv(output / "filter_summary.csv", summary)
    for noise in ("salt_pepper", "gaussian"):
        best = max((row for row in summary if row["noise"] == noise and row["method"] != "noisy"),
                   key=lambda row: row["mean_psnr_db"])
        print(f'{noise}: {best["method"]} filter, mean PSNR {best["mean_psnr_db"]:.2f} dB')
    return panels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "images", help="Image file or directory")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--impulse-amount", type=float, default=0.05)
    parser.add_argument("--gaussian-sigma", type=float, default=20.0)
    parser.add_argument("--show", action="store_true", help="Display panels in an OpenCV window")
    args = parser.parse_args()
    if not 0 <= args.impulse_amount <= 1 or not math.isfinite(args.gaussian_sigma) or args.gaussian_sigma < 0:
        parser.error("Use an impulse amount between 0 and 1 and a finite, nonnegative Gaussian sigma.")
    if args.seed < 0:
        parser.error("Use a nonnegative seed.")
    paths = sorted(path for path in args.input.iterdir() if path.is_file()
                   and path.suffix.lower() in IMAGE_EXTENSIONS) if args.input.is_dir() else [args.input]
    if not paths or not args.input.exists():
        parser.error(f"No images found at {args.input}")
    try:
        panels = run(paths, args.output, args.seed, args.impulse_amount, args.gaussian_sigma)
    except (ValueError, OSError, cv2.error) as exc:
        parser.exit(1, f"{exc}\n")
    print(f"Saved {len(panels)} comparison panels to {args.output}")
    if args.show:
        for panel in panels:
            cv2.imshow("Denoising | any key to advance, Esc to exit", cv2.imread(str(panel)))
            if cv2.waitKey(0) == 27:
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
