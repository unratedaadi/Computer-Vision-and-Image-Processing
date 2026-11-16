# Smart Image Enhancement and Quality Analysis

The system flagged all five controlled inputs according to their assigned test conditions. A different enhancement gave the highest structural similarity on each of the three images that improved. The original test input scored highest on the other two.

## Experiment setup

Five public scikit-image sample photographs supplied the clean references. Each test input was produced from its reference by applying the specified gain and offset to the 8-bit luma channel, then rounding and clipping to 0–255. This creates known exposure and contrast changes while keeping the scene aligned for pixel-level evaluation. Source details and credits are available in the [scikit-image sample data documentation](https://scikit-image.org/docs/stable/api/skimage.data.html).

| Scene | Assigned condition | Luma gain | Luma offset | Source credit |
| --- | --- | ---: | ---: | --- |
| Astronaut | Darkened | 1.00 | −48 | NASA Great Images database |
| Brick | Compressed contrast | 0.42 | +70 | scikit-image sample data |
| Camera | Compressed contrast | 0.40 | +77 | scikit-image sample data |
| Coins | Darkened | 1.00 | −52 | Brooklyn Museum Collection |
| Rocket | Darkened and compressed contrast | 0.57 | +18 | SpaceX Photos |

OpenCV loaded each input into BGR and converted it to YCrCb for processing. The program flags low brightness when mean luma is below 90 and low contrast when the P95–P05 luma span is below 110. These cutoffs are demonstration heuristics. In particular, a naturally dark scene can trigger a brightness flag.

Each algorithm applied one fixed configuration across all five samples. Brightness adjustment adds 45 to luma with saturation. Contrast stretching maps the 2nd and 98th percentiles to 0 and 255 and clips outside that range. Histogram equalization remaps luma through the cumulative intensity distribution using OpenCV. The chroma channels are carried through the conversion. Extreme luma changes can still clip BGR values during reconstruction.

The histograms show the percentage of image pixels at each luma level. Mean luma estimates brightness, and luma standard deviation and the P95–P05 span describe global contrast. The CSV also reports the shares of pixels at luma ≤15 and ≥250. PSNR measures pixel fidelity in decibels, and SSIM measures similarity in local luminance, contrast, and structure. Both use the clean reference and all three color channels. This study uses higher SSIM as its primary criterion, with PSNR reported as a second measure. A comparison without a reference uses the histogram and luma statistics.

## Measured results

Each result cell gives **SSIM / PSNR in dB**. Bold marks the highest SSIM in each row, with the original input eligible. See [metrics.csv](results/metrics.csv) for the full precision and luma statistics.

| Scene and comparison | Original input | Brightness adjustment | Contrast stretching | Histogram equalization |
| --- | ---: | ---: | ---: | ---: |
| [Astronaut](results/astronaut/comparison.png) | 0.781 / 15.84 | 0.789 / 21.96 | 0.881 / 23.45 | **0.897 / 24.13** |
| [Brick](results/brick/comparison.png) | **0.875 / 24.03** | 0.812 / 13.71 | 0.538 / 11.93 | 0.443 / 12.67 |
| [Camera](results/camera/comparison.png) | 0.730 / 15.22 | 0.692 / 12.18 | **0.958 / 21.18** | 0.833 / 21.89 |
| [Coins](results/coins/comparison.png) | 0.579 / 14.34 | **0.950 / 29.80** | 0.639 / 17.78 | 0.671 / 17.73 |
| [Rocket](results/rocket/comparison.png) | **0.945 / 24.31** | 0.821 / 16.29 | 0.886 / 19.15 | 0.629 / 9.62 |

## Why the results differ

- **Astronaut.** Equalization gave the best reference match on both metrics. It lifted midtones while retaining more dark tones than the fixed brightness offset.
- **Brick.** The clean surface has a narrow tonal range. Stretching expanded the input's P95–P05 span from 34 to 217 and exaggerated the tonal differences. Brightness adjustment was closest among the three processed versions by SSIM, but it still reduced fidelity.
- **Camera.** Stretching recovered the tonal separation between the subject and background and had the highest SSIM. Equalization scored 0.71 dB higher in PSNR. The two metrics rank this image differently.
- **Coins.** A fixed luma increase partially reversed the darkening and preserved the P95–P05 span of 139, whereas stretching and equalization expanded it beyond 225. Brightness adjustment led both reference metrics.
- **Rocket.** The night sky is deliberately dark in the clean reference. The processed versions lift the shadows too much or exaggerate the launch lights. Stretching came closest among the enhanced versions by SSIM, and the original input remained closest overall.

This benchmark uses five scenes and simulated global changes. It does not measure motion blur, sensor noise, or uneven illumination. Darkening can clip shadow levels into a single value, making them unrecoverable using a global remap. A follow-up experiment could compare gamma correction for underexposed photos and CLAHE for uneven lighting, using real camera exposures with matching clean references where possible.

Benchmark run with Python 3.13, OpenCV 4.13.0, NumPy 2.3.5, Matplotlib 3.10.6, and scikit-image 0.25.2.
