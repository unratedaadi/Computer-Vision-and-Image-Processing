# Real-Time Object Detection and Counting System

**Mini Project 7 · Computer Vision and Image Processing**  
**Aaditya Nahar Jain · B.Tech in Computer Science and Engineering (AI and Data Engineering)**  
Jain (Deemed-to-be University), Faculty of Engineering and Technology

## Objective

Locate and count objects in camera frames, video, and photographs. Compare a traditional person detector against a pretrained deep-learning detector using the same manually annotated person instances. Image classification assigns labels to the whole image. Object detection predicts a class and a bounding box for each visible instance, which allows multiple objects of the same class to be counted.

## Implementation

1. OpenCV loads each image or decodes each video frame as BGR pixels.
2. The traditional branch extracts a histogram of oriented gradients (HOG) in sliding windows at multiple scales. OpenCV's linear SVM predicts whether a window contains a person and groups nearby detections. The label displays the SVM margin, which measures classifier separation.
3. The deep-learning branch resizes each frame to 320 × 320, converts BGR to RGB, and scales pixel values by 1/255. OpenCV DNN loads pretrained YOLO-Fastest 1.1 Darknet weights. It filters detections at a score of 0.25 and uses class-aware non-maximum suppression at IoU 0.45. Its COCO label set contains 80 categories.
4. The program draws the retained boxes and scores and counts the boxes per category. The CSV output preserves counts by image or frame.

## Test design

The five original photos cover pedestrians beside a bus, overlapping people, an indoor sports scene, a cyclist and photographer, and a street image with no people. Three derivatives simulate underexposure, smaller objects, and a dense frame. Underexposure multiplies the bus image pixel values by 0.14. The smaller-object case reduces the court image to 25% of its original dimensions. The crowded montage places four resized photographs on one canvas.

The person annotations include clearly identifiable foreground people and identifiable figures cut off at the image boundary. Indistinct figures in the blurred stadium background were excluded. The annotation file uses pixel coordinates in `[x, y, width, height]` format. The same reference boxes are transformed with their images in the derived cases.

Predictions are matched in descending score order. A person prediction counts as a true positive if its intersection-over-union with an unmatched reference box is at least 0.50. Unmatched predictions are false positives. Unmatched references are false negatives. Precision is TP/(TP+FP), recall is TP/(TP+FN), and F1 is their harmonic mean. Count mean absolute error averages the per-image absolute difference between predicted and reference person counts.

## Results

| Detector | TP | FP | FN | Precision | Recall | F1 | Person count MAE | Median ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HOG + linear SVM | 2 | 9 | 24 | 0.1818 | 0.0769 | 0.1081 | 2.125 | 149.635 |
| YOLO-Fastest 1.1 | 17 | 7 | 9 | 0.7083 | 0.6538 | 0.68 | 1 | 12.762 |


The benchmark ran on an Apple Silicon Mac with OpenCV 4.13.0 and one CPU thread. Each image was run five times after warm-up, and the median per image was used. The table reports the median of those image medians. Model initialization, reading files, drawing overlays, and writing output are outside the measured detector latency.

| Method | Category coverage | Main observed limitation |
|---|---|---|
| HOG + linear SVM | Person | Sliding-window boxes often failed to align with cropped, overlapping, or seated people. |
| YOLO-Fastest 1.1 | 80 COCO labels | Small people were missed and the crowded montage produced several duplicate or poorly localized detections. |

### Failure cases

- **Overlap.** YOLO matched both foreground people in the team photograph. HOG found one candidate, but its box overlapped neither reference at the required threshold.
- **Low light.** YOLO detected three of four people in the darkened bus image, compared with two of four in the original. HOG returned one candidate in the darkened image and none matched the reference boxes. A single simulated exposure change cannot establish robustness in nighttime footage.
- **Small objects.** Both methods scored zero true positives when the court photo was downscaled to 25%. YOLO returned zero people and HOG returned one poorly localized candidate.
- **Crowded frame.** YOLO reported 12 people for 10 annotated instances, with six true positives. HOG reported two people and matched one. The montage tests many small instances in one frame. A campus crowd would add natural occlusion and motion.
- **Background and class confusion.** Both methods reported one person in the street photo with no people. YOLO also predicted a tennis racket in the court scene. Only the person category has reference boxes, so the other category totals are descriptive.

### Conclusion

On this sample, YOLO located more people, made smaller per-image counting errors, and took less CPU time than the HOG baseline. The benchmark includes eight cases drawn from five original photographs, and the derived images share source content. The person results describe this set and should not be interpreted as a population-level accuracy estimate. The video demo verifies the frame-processing pipeline with a slideshow. Physical webcam performance and unique-object traffic counts were outside the measured evaluation.

## Demonstration and viva

Demonstrate the annotated video first. Then run one image through both methods, point to the person boxes and count rows, and compare the original court image with the reduced one. Open `results/comparison.md` to explain how the counts and localization scores differ.

| Viva question | Answer |
|---|---|
| How does YOLO identify multiple objects? | One network pass produces candidate boxes and class scores at different spatial locations. Filtering and suppression yield the final detections. |
| What does the confidence threshold change? | A higher threshold removes more low-score boxes. It can reduce false positives and can also suppress true objects. |
| What is IoU? | Intersection area divided by union area for a pair of boxes. This project uses 0.50 for scoring and 0.45 for YOLO's duplicate suppression. |
| Why can a correct count have low F1? | A missed person and a false positive can cancel out in the total. F1 also checks the location of each predicted box. |
| Why does HOG struggle with distant or cropped people? | Small objects provide weak gradient detail, and cropped people lack the upright shape the sliding window expects. |
| How would you count people entering a room only once? | Track detections with stable IDs and record a crossing when an ID moves across a virtual line in the entry direction. |
| Did you train YOLO? | The project uses pretrained COCO weights and evaluates them on the local test images. |

## References and image credits

- HOG + SVM detector API and parameters. [OpenCV HOG documentation](https://docs.opencv.org/5.0/extra_modules/xobjdetect_hog.html)
- Pretrained YOLO-Fastest 1.1 weights, model configuration, and label map. [Upstream COCO model directory](https://github.com/dog-qiuqiu/Yolo-Fastest/tree/master/ModelZoo/yolo-fastest-1.1_coco)
- Bus photo. [Ultralytics sample image](https://github.com/ultralytics/assets/blob/main/im/bus.jpg)
- Team photo. [Ultralytics Zidane sample image](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/assets/zidane.jpg)
- Court photo. [OpenCV basketball sample](https://github.com/opencv/opencv/blob/4.x/samples/data/basketball1.png)
- Cyclist photo. [YOLO-FastestV2 sample image](https://github.com/dog-qiuqiu/Yolo-FastestV2/blob/main/img/000139.jpg)
- Street photo. [Darknet dog sample image](https://github.com/pjreddie/darknet/blob/master/data/dog.jpg)
