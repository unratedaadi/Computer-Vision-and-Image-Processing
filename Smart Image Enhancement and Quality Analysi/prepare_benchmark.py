"""Reproduce the bundled degraded inputs from the five clean public photos."""

import cv2
import numpy as np

from enhance import ROOT, read_image, with_luma, write_png

# For each reference, transform Y with round(clip(gain * Y + offset)).
CASES = {
    "astronaut.png": (1.0, -48),
    "brick.png": (0.42, 70),
    "camera.png": (0.40, 77),
    "coins.png": (1.0, -52),
    "rocket.jpg": (0.57, 18),
}


def main():
    destination = ROOT / "images/input"
    destination.mkdir(parents=True, exist_ok=True)
    for name, (gain, offset) in CASES.items():
        reference_path = ROOT / "images/reference" / name
        reference = read_image(reference_path)
        ycrcb = cv2.cvtColor(reference, cv2.COLOR_BGR2YCrCb)
        source = ycrcb[:, :, 0].astype(np.float32)
        degraded = np.rint(np.clip(gain * source + offset, 0, 255)).astype(np.uint8)
        write_png(destination / f"{reference_path.stem}.png", with_luma(ycrcb, degraded))
    print(f"Prepared {len(CASES)} inputs in {destination}")


if __name__ == "__main__":
    main()
