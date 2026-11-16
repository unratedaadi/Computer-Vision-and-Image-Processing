# Mini Project 7 · Real-Time Object Detection and Counting System

**Aaditya Nahar Jain** · Computer Vision and Image Processing

This Python project uses OpenCV to detect and count objects in images, video files, and webcam frames. The pretrained COCO YOLO-Fastest 1.1 model and a traditional HOG + linear SVM person detector are both included. Each output shows bounding boxes, classes, detection scores, per-category counts, and detector latency.

## Setup

Open a terminal in the extracted folder. Use Python 3.11–3.13.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with `.\.venv\Scripts\Activate.ps1`. The pretrained weights are bundled, so inference works offline after installing the dependencies.

## Demonstration

```bash
python detect.py inputs/demo.mp4 --show --save-video --output run/demo
python detect.py inputs --method yolo --output run/yolo
python detect.py inputs --method hog --output run/hog
python detect.py 0 --show --output run/webcam
```

The 10-second demo clip is a slideshow of test photos. The video command runs detection on each of its 120 frames. Press **Q** or **Esc** to close the preview. Remove `--show` for headless video processing. Webcam mode needs an available camera and OS permission. Open `results/demo/annotated.mp4` to see the recorded output immediately.

Each frame count describes what the detector sees at that instant. An object can be counted again in the next frame because the project does not assign tracking IDs.

## Project files

| File or folder | Contents |
|---|---|
| `detect.py` | Both detectors, image/video/webcam processing, annotations, and count export |
| `prepare_demo.py` | Rebuilds the derived images, person reference boxes, and demo clip |
| `benchmark.py` | Scores boxes at IoU ≥ 0.50 and measures processing time |
| `inputs/` | Five public photos, three derived cases, demo video, and hand-marked person boxes |
| `weights/` | Pretrained YOLO weights, Darknet configuration, COCO labels, and upstream license |
| `results/hog/`, `results/yolo/` | Annotated still images for both detectors |
| `results/object_counts.csv` | Detected counts per method, image, and class |
| `results/category_totals.csv` | Category counts summed across cases |
| `results/comparison.md` | Method comparison and per-image counts |
| `REPORT.md` | Methodology, measured results, failure analysis, and viva preparation |

Reproduce the benchmark from the original images with:

```bash
python prepare_demo.py
python benchmark.py
```
