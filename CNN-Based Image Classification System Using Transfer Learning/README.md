# DigitLab

**CVIP Mini Project 9 · CNN-Based Image Classification System Using Transfer Learning**

DigitLab classifies isolated handwritten digits from 0 to 9. It compares a basic CNN trained from scratch with an ImageNet-pretrained ResNet18 adapted to the digit classes.

## Run the demo

Use Python 3.11 or newer on a platform supported by the pinned PyTorch release.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:8765/. Choose a held-out sample, pick a model, and select **Classify image**. You can also upload a tightly cropped PNG, JPEG, or WebP containing a single digit. Use **Invert** for dark ink on a light background. The weights are included, so the demo needs no network after dependency installation.

You can run the same inference pipeline from the terminal.

```bash
python app.py --sample 8
python app.py --sample 2 --model cnn
python app.py --image my-digit.png --invert
```

## Dataset and preprocessing

Scikit-learn ships 1,797 grayscale digit images from the UCI Optical Recognition of Handwritten Digits collection. The source rasters are 8 × 8 with integer values from 0 to 16. The supplied subset corresponds to the original UCI test partition. This project makes its own seeded, stratified 70/15/15 split within that subset, grouping any exactly identical rasters. Writer identifiers are unavailable in this subset, so the split cannot establish generalization to unseen writers.

| Split | Images | Use |
|---|---:|---|
| Train | 1,257 | Fit both models |
| Validation | 270 | Choose a checkpoint by minimum cross-entropy |
| Test | 270 | Measure final metrics and inspect errors |

Both models use bilinear resizing to 64 × 64. The basic CNN uses a grayscale channel scaled to [0, 1]. ResNet18 uses three copies of the grayscale channel followed by the ImageNet channel means and standard deviations. The same preprocessing function is used for test images and demo uploads.

## Reproduce the experiment

```bash
python train.py
python train.py --weights /path/to/model.safetensors
```

Run one of the commands above. The first command fetches the pinned ImageNet checkpoint from Hugging Face. The second accepts its local copy. Training replaces the experiment outputs. The recorded run used seed 42, CPU, four PyTorch threads, batch size 64, and AdamW with cross-entropy. The basic CNN trained for eight epochs. ResNet18 trained its new ten-class head for four epochs, then fine-tuned its last residual stage and head for four more epochs with batch-normalization layers frozen. Both checkpoints were selected by validation loss.

## Results

| Held-out metric | Basic CNN | ResNet18 |
|---|---:|---:|
| Accuracy | 92.22% | 94.81% |
| Macro precision | 92.14% | 94.81% |
| Macro recall | 92.08% | 94.84% |
| Macro F1 | 91.77% | 94.77% |
| Dim images at 40% brightness | 74.07% | 90.74% |
| Textured background | 70.00% | 61.48% |

ResNet18 classified seven more clean test images correctly. Its worst class by clean recall was digit 9 (22/27). The basic CNN struggled most with digit 8 (15/26). The test support ranges from 26 to 28 examples per class, so class frequency alone does not explain the errors. The texture experiment uses a controlled synthetic background applied to the same test rasters. The result describes this specific shift. Both models can assign high scores to wrong predictions.

## Submission files

- `DigitLab_CVIP_Report.pdf` covers methods, results, error analysis, limitations, and viva questions.
- `classifier.py`, `train.py`, `app.py` and `demo/` contain model, experiment, and demo source code.
- `outputs/training_curves.png` and `outputs/confusion_matrices.png` contain the comparison figures.
- `outputs/sample_predictions.png` and `outputs/error_examples.png` show held-out examples and actual model errors.
- `outputs/metrics.json` includes per-class precision, recall, F1 and support, split counts, histories, and checkpoint provenance.
- `outputs/splits.csv` and `outputs/test_predictions.csv` provide the split manifest and row-level predictions.
- `outputs/cnn.pt` and `outputs/resnet.pt` are the trained model weights.

## Sources

- [Optical Recognition of Handwritten Digits, UCI Machine Learning Repository](https://doi.org/10.24432/C50P49)
- [Scikit-learn dataset documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
- [ImageNet ResNet18 checkpoint and model card](https://huggingface.co/timm/resnet18.a1_in1k)
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

The sample images are grayscale exports from UCI Optdigits, attributed to E. Alpaydin and C. Kaynak. The ResNet checkpoint adapts ImageNet weights from the timm model card linked above. Those pretrained weights are published under Apache 2.0, reproduced in `APACHE-2.0.txt`.
