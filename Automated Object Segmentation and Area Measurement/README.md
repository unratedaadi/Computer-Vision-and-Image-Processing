# Mini Project 5 — Automated Object Segmentation and Area Measurement

This Python project compares fixed global thresholding with Otsu thresholding
for locating leaves, counting connected regions and measuring their areas in
square pixels. It includes five real scans, derived test cases, reference masks,
full OpenCV outputs and an offline report with an interactive case selector.

## Open the demo

Extract the complete folder and open **report.html** in a browser. Select a case
to compare its input, manual reference mask, two predicted masks, annotated
images, counts, areas and IoU. The report also contains the experimental
method, aggregate results and failure analysis.

## Run the pipeline

Use Python 3.10 or newer. From the extracted project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python segment.py
python build_report.py
```

On Windows PowerShell activate with `.venv\Scripts\Activate.ps1` after creating
the environment. With Anaconda and an existing Python that imports OpenCV and
NumPy, run the last two commands directly in that environment.

Display the generated comparisons in an OpenCV window with
`python segment.py --show`. Press **N** or **Space** to advance, **P** to go
back, and **Q** or **Esc** to exit. The browser report provides the same
comparisons through the case selector.

For your own photo or folder:

```bash
python segment.py --input path/to/photos --polarity dark --output my-results
```

Use `--polarity light` for bright objects against a dark background. Adjust
`--threshold`, `--kernel` and `--min-area` for the contrast and scale of your
own images. Optional `--reference path/to/masks` accepts binary PNG reference
masks with the same filename stem and dimensions as their inputs. Custom
images are processed at their decoded image size.

## Project files

| File or directory | Contents |
| --- | --- |
| `segment.py` | Segmentation, measurement, metrics and optional OpenCV viewer |
| `prepare_dataset.py` | Regenerates benchmark inputs and manual reference masks from the source scans |
| `build_report.py` | Regenerates the offline report from the benchmark CSV |
| `data/originals/` | Five source JPEG scans at their supplied resolution |
| `data/inputs/` | Five resized scans, ten simulated lighting variants and two three-leaf composites |
| `data/reference/` | Same-size binary reference silhouettes |
| `data/cases.json` | Scenario metadata, source filenames and expected counts |
| `outputs/summary.csv` | Counts, areas, thresholds and reference comparisons per image and method |
| `outputs/objects.csv` | Area, bounding box and centroid per detected region |
| `outputs/<case>/` | Grayscale image, raw and cleaned masks, annotated outputs and comparison view |
| `report.html` | Experimental report and browser demo |
| `VIVA.md` | Walkthrough and sample viva questions with answers |

To regenerate the benchmark images before running the pipeline, run
`python prepare_dataset.py`. Areas are tied to the benchmark input resolution.
The source scans are 2549 × 3500, with single-leaf evaluation at 465 × 638.

## Dataset credit

Source scans are from John Lagergren et al., *Few-Shot Learning Enables
Population-Scale Analysis of Leaf Traits in Populus trichocarpa*, Plant
Phenomics, DOI [10.34133/plantphenomics.0072](https://doi.org/10.34133/plantphenomics.0072).
The five JPEG filenames match the [source repository](https://github.com/jlager/few-shot-leaf-segmentation/tree/main/data/images).
The repository's GPL-3.0 license text is included in
[`data/SOURCE_LICENSE.md`](data/SOURCE_LICENSE.md). The benchmark uses project
annotations traced on the scans. The simulated tests and composite scenes
are documented in the report.
