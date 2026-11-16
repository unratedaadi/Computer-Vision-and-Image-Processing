"""Rebuild the stress-test images, person annotations, and demonstration clip."""
import json
from pathlib import Path

import cv2
import numpy as np

from detect import read_image, save_image


ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "inputs"
SOURCES = {
    "01_bus.jpg": {
        "condition": "Daylight; walking people and two truncated figures",
        "people": [[0, 515, 72, 365], [48, 389, 179, 515],
                   [225, 400, 121, 455], [686, 385, 124, 490]],
    },
    "02_team.jpg": {
        "condition": "Overlapping, cropped foreground people",
        "people": [[120, 199, 1035, 521], [744, 40, 397, 680]],
    },
    "03_court.png": {
        "condition": "Indoor grayscale; full-body and cropped people",
        "people": [[26, 79, 132, 397], [419, 14, 221, 466]],
    },
    "04_cyclist.jpg": {
        "condition": "Riding pose and partially visible photographer",
        "people": [[148, 54, 118, 160], [0, 167, 101, 167]],
    },
    "05_street.jpg": {
        "condition": "No people; bicycle, truck and dog",
        "people": [],
    },
}


def resized_boxes(boxes, scale, offset_x=0, offset_y=0):
    return [
        [round(x * scale) + offset_x, round(y * scale) + offset_y,
         round(w * scale), round(h * scale)]
        for x, y, w, h in boxes
    ]


def letterbox(frame, width, height):
    h, w = frame.shape[:2]
    scale = min(width / w, height / h)
    resized = cv2.resize(frame, (round(w * scale), round(h * scale)),
                         interpolation=cv2.INTER_AREA)
    x, y = (width - resized.shape[1]) // 2, (height - resized.shape[0]) // 2
    canvas = np.full((height, width, 3), 30, dtype=np.uint8)
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas, scale, x, y


def main():
    originals = {name: read_image(INPUTS / name) for name in SOURCES}
    bus = originals["01_bus.jpg"]
    dark = np.rint(bus.astype(np.float32) * 0.14).astype(np.uint8)
    save_image(INPUTS / "06_bus_low_light.png", dark)

    court = originals["03_court.png"]
    small = np.full_like(court, 30)
    small[180:300, 240:400] = cv2.resize(court, (160, 120),
                                        interpolation=cv2.INTER_AREA)
    save_image(INPUTS / "07_court_small.png", small)

    cells = ["01_bus.jpg", "02_team.jpg", "03_court.png", "04_cyclist.jpg"]
    tiles = []
    crowd_people = []
    for position, name in enumerate(cells):
        tile, scale, x, y = letterbox(originals[name], 320, 240)
        tiles.append(tile)
        crowd_people.extend(resized_boxes(
            SOURCES[name]["people"], scale,
            x + 320 * (position % 2), y + 240 * (position // 2),
        ))
    crowd = np.vstack((np.hstack(tiles[:2]), np.hstack(tiles[2:])))
    save_image(INPUTS / "08_crowded_montage.png", crowd)

    cases = {
        **SOURCES,
        "06_bus_low_light.png": {
            "condition": "Simulated low light; 14% source intensity",
            "people": SOURCES["01_bus.jpg"]["people"],
        },
        "07_court_small.png": {
            "condition": "Simulated distance; actors scaled to 25%",
            "people": resized_boxes(SOURCES["03_court.png"]["people"], .25, 240, 180),
        },
        "08_crowded_montage.png": {
            "condition": "Synthetic crowded montage; 10 people across four panels",
            "people": crowd_people,
        },
    }
    (INPUTS / "person_ground_truth.json").write_text(json.dumps(cases, indent=2) + "\n")

    clip = cv2.VideoWriter(str(INPUTS / "demo.mp4"),
                           cv2.VideoWriter_fourcc(*"mp4v"), 12, (768, 576))
    if not clip.isOpened():
        raise OSError("Unable to create inputs/demo.mp4")
    try:
        for name in ("01_bus.jpg", "03_court.png", "05_street.jpg",
                     "06_bus_low_light.png", "08_crowded_montage.png"):
            frame, _, _, _ = letterbox(read_image(INPUTS / name), 768, 576)
            for _ in range(24):
                clip.write(frame)
    finally:
        clip.release()


if __name__ == "__main__":
    main()
