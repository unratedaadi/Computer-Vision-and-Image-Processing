# Mini Project 1 · Smart Image Enhancement and Quality Analysis

This Python and OpenCV project detects low brightness and low contrast, applies three enhancement methods, and compares their effects using luma histograms and quantitative metrics.

## Run the project

Use Python 3.10 or newer. From this folder, install the dependencies and run the bundled benchmark.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python enhance.py
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1` and use `python` to create the environment. If your existing Conda environment already has the dependencies, run `python enhance.py` there.

```bash
# Display comparisons in OpenCV windows. Press any key for the next image.
python enhance.py --show

# Process your own JPEG or a folder of photos.
python enhance.py /path/to/photo.jpg --output results/my_photo
python enhance.py /path/to/photo_folder --output results/my_photos

# Supply clean references with matching filename stems and dimensions.
python enhance.py /path/to/inputs --reference-dir /path/to/references --output results/custom

# Recreate the five degraded inputs from the supplied clean reference files.
python prepare_benchmark.py
```

## Deliverables

| Path | Contents |
| --- | --- |
| `enhance.py` | Image loading, problem detection, enhancement, previews, and evaluation |
| `prepare_benchmark.py` | Reproducible degradation script |
| `images/reference/` | Five clean, publicly available scikit-image sample photos in their distributed formats |
| `images/input/` | Five controlled test inputs saved as PNG |
| `results/<image>/` | Three enhanced PNGs and an illustrated comparison with overlaid histograms |
| `results/metrics.csv` | All 20 measured rows, including the five input baselines |
| [RESULTS.md](RESULTS.md) | Experiment setup, results table, best method per image, and discussion |

The pipeline decodes supported files to 8-bit, three-channel BGR through OpenCV. Grayscale photos appear with neutral color channels. Outputs are lossless PNGs. PSNR and SSIM are left blank for inputs without a matching clean reference. On images smaller than 3 pixels in either dimension, SSIM is unavailable.
