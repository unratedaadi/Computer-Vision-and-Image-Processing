# Demonstration and viva

## Three-minute demonstration

1. Open `report.html` and choose **Leaf 04 · Original lighting**. Identify the
   input, reference, masks and overlays. Point to the threshold and area in
   the measurement table.
2. Choose **Leaf 04 · Simulated dim light**. The fixed method returns a count
   of one because it labels all 296,670 pixels as foreground. Compare its
   area and IoU to Otsu's output.
3. Choose **Leaf 04 · Simulated side shadow**. Show the dark background band
   included by both masks. Explain why a single threshold cannot distinguish
   a shadowed background from the leaf.
4. Switch from **Separated composite** to **Touching composite**. The same
   three source leaves go from three detected components to one as they overlap.
5. Run `python segment.py` and open `outputs/objects.csv`. Show the area of
   each individual detected region and how these add to the summary area.

## Questions and answers

| Question | Answer |
| --- | --- |
| What is image segmentation? | It assigns a label to each pixel. This project uses two labels, foreground leaf and background. |
| Why convert to grayscale and blur? | Grayscale gives one intensity value per pixel. A 5 × 5 Gaussian blur reduces small intensity fluctuations before thresholding, at the expense of some edge detail. |
| How does fixed global thresholding work? | With dark foreground, pixels at or below 200 are assigned 255 and all others are assigned 0. The same value is used for every benchmark image. |
| What does Otsu's method choose? | It picks a single threshold from the current image histogram by maximizing between-class variance, equivalent to minimizing weighted within-class variance. |
| Is Otsu locally adaptive? | It adapts its threshold between images. Every pixel in a given image uses the same threshold. |
| What do opening and closing do? | Opening erodes then dilates to remove small foreground features. Closing dilates then erodes to fill narrow gaps and small holes. Both can affect edges and area. |
| Why use connected components? | Each 8-connected region is counted once. The component's pixel count is its projected area. Eight-connectivity includes diagonal neighbors. |
| What is IoU? | The number of pixels shared by prediction and reference divided by the number covered by either. A score of 1 means identical masks. |
| Can a correct count hide a poor mask? | Yes. In the dim case, a single connected component can cover the entire image and still match the expected count of one. |
| Why do overlapping objects get merged? | Connected components treat touching foreground pixels as one region. Watershed can separate some touching objects using markers. Heavy occlusion may require an instance model and cannot reveal the exact hidden area. |
| How would you report area in mm²? | Place a calibration target on the same plane and estimate the scale in mm/pixel. Multiply pixel area by the square of that scale under uniform scale and a fronto-parallel view. |
| How would you improve the shadow case? | Try more even lighting, background correction or local thresholding and validate against fresh annotated images. |
| What limits the conclusion? | The benchmark has five related source scans, manual polygon references and simulated stress tests. Real camera photos and more independent samples would strengthen the comparison. |
