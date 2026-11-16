# Mini Project 3 | Image Denoising and Quality Enhancement System

## Run and demonstrate

From this folder, use Python 3.10 or newer with a graphical desktop for the live demo.

```bash
python -m pip install -r requirements.txt
python denoise.py --show
```

Press any key to advance through the comparisons and Esc to close the viewer. Run `python denoise.py` to regenerate the PNGs and CSV files without opening windows. Run `python denoise.py --input photo.jpg --output custom_outputs --show` to apply the same experiment to a clean reference photo of your own.

1. On the **Chelsea** salt-and-pepper panel, point out the black and white speckles in the noisy image. Compare the speckles remaining after the linear filters with the cleaner median result. Look at the whiskers and fur to spot detail lost to smoothing.
2. On the **Coffee** Gaussian panel, compare the grain on the wood and the texture along the cup rim. The Gaussian filter suppresses the grain, and some fine wood texture softens.
3. Open `outputs/filter_summary.csv` and explain how the PSNR values support each conclusion. The full-resolution outputs are alongside the smaller preview panels.

## Experiment

The five public scikit-image 0.25.2 images are NASA's Eileen Collins astronaut portrait, Lav Varshney's Camera (CC0), Stefan van der Walt's Chelsea (CC0), Rachel Michetti's Coffee (CC0), and SpaceX's DSCOVR rocket launch (public domain). [Image source and credits](https://scikit-image.org/docs/stable/api/skimage.data.html).

Each image is loaded with OpenCV at its native resolution. NumPy uses seed 42 and selects 5% of pixel positions without replacement for salt-and-pepper noise, with the positions split between white and black. Gaussian noise is sampled independently at each pixel and channel, with mean 0 and standard deviation 20 intensity levels. The result is rounded and clipped to the 8-bit range. All three filters receive the same noisy image for a given test, use a 3×3 neighborhood, and replicate border pixels. The Gaussian filter uses sigma 0.85.

MSE is the average squared difference from the clean reference over all pixels and channels. PSNR is `10 × log10(255² / MSE)`, with an exact match giving infinite PSNR. Lower MSE and higher PSNR indicate closer agreement with the reference. The summary averages per-image MSE and per-image PSNR independently, giving each image equal weight.

## Measured results

| Noise | Method | Mean MSE | Mean PSNR (dB) |
|---|---|---:|---:|
| Salt-and-pepper | Noisy input | 1062.79 | 17.88 |
| Salt-and-pepper | Mean | 198.24 | 25.23 |
| Salt-and-pepper | Gaussian | 206.98 | 25.02 |
| Salt-and-pepper | **Median** | **55.86** | **30.91** |
| Gaussian | Noisy input | 371.40 | 22.44 |
| Gaussian | Mean | 110.62 | 27.77 |
| Gaussian | **Gaussian** | **98.79** | **28.23** |
| Gaussian | Median | 123.20 | 27.26 |

The median filter ranked first on all five salt-and-pepper images. The Gaussian filter ranked first on four of the five Gaussian-noise images. The mean filter ranked first for Chelsea. For salt-and-pepper noise, the median rejects isolated extreme values by choosing the middle-ranked value in each neighborhood. Averaging spreads an impulse's effect to nearby output pixels. For Gaussian noise, averaging suppresses small random fluctuations. The Gaussian weights favor pixels nearest the center and produced the best mean PSNR at the tested setting, 0.45 dB ahead of the mean filter. These rankings describe five images at the tested strengths and kernel size. Denoising can erase fine texture, and PSNR alone does not measure perceived sharpness.

`outputs/per_image_metrics.csv` contains 40 rows. Each image folder contains its clean reference, both noisy versions, their six filtered outputs, and two labeled comparison panels.

## Viva preparation

| Question | Answer |
|---|---|
| Why are mean and Gaussian filtering linear? | Each output uses a weighted sum of the neighborhood. The mean uses equal weights and the Gaussian uses weights that decrease with distance. This describes the filtering operation before 8-bit rounding. |
| Why is the median filter nonlinear? | It sorts values and selects the middle rank. That selection cannot be represented by a fixed weighted sum. |
| What happens if the kernel grows? | A larger neighborhood can reduce more noise, while increasing the chance of flattening edges and removing fine structures. |
| Why add noise to an original image? | A known reference makes it possible to calculate reconstruction error and give each filter the same input. |
| Why convert to floating point before computing MSE? | Subtracting two unsigned 8-bit pixels can wrap around and give the wrong difference. |
| Can you report PSNR on a single noisy camera photo? | Conventional PSNR requires a matching clean reference. A naturally noisy photo can still be compared visually. |
| What is a limitation of per-channel median filtering? | It can pick channel values from different neighboring pixels and introduce color artifacts. |
