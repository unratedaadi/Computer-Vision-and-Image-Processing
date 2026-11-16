"""Compare a scratch CNN and ImageNet transfer learning on UCI optical digits."""

import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "digitlab-mpl"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageEnhance
from sklearn import __version__ as sklearn_version
from sklearn.datasets import load_digits
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from classifier import CLASSES, MODEL_NAMES, ROOT, build_model, digit_image, preprocess

SEED = 42
OUT = ROOT / "outputs"
WEIGHT_REPO = "timm/resnet18.a1_in1k"
WEIGHT_REVISION = "491b427b45c94c7fb0e78b5474cc919aff584bbf"


def split_indices(pixels, labels):
    # Group exact duplicate rasters by signature before splitting to prevent leakage.
    _, first, groups = np.unique(pixels.reshape(len(pixels), -1), axis=0, return_index=True, return_inverse=True)
    group_ids = np.arange(len(first))
    train_val, test = train_test_split(group_ids, test_size=0.15, random_state=SEED, stratify=labels[first])
    train, val = train_test_split(train_val, test_size=0.15 / 0.85, random_state=SEED, stratify=labels[first[train_val]])
    return {name: np.flatnonzero(np.isin(groups, ids)) for name, ids in (
        ("train", train), ("validation", val), ("test", test)
    )}


def make_loader(images, labels, indices, model_name, shuffle=False):
    x = torch.stack([preprocess(images[i], model_name) for i in indices])
    y = torch.as_tensor(labels[indices], dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=64, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(SEED))


def epoch_pass(model, loader, device, optimizer=None, transfer=False):
    model.train(optimizer is not None)
    if optimizer is not None and transfer:
        for layer in model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
    loss_total = 0.0
    scores, targets = [], []
    criterion = nn.CrossEntropyLoss()
    with torch.set_grad_enabled(optimizer is not None):
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            loss_total += loss.item() * len(labels)
            scores.append(logits.detach().softmax(1).cpu())
            targets.append(labels.cpu())
    scores, targets = torch.cat(scores), torch.cat(targets)
    return loss_total / len(targets), (scores.argmax(1) == targets).float().mean().item(), scores.numpy()


def train_model(model_name, loaders, args, device, weights):
    torch.manual_seed(SEED)
    model = build_model(model_name, weights if model_name == "resnet" else None).to(device)
    if model_name == "resnet":
        model.requires_grad_(False)
        model.fc.requires_grad_(True)
        stages = [("Frozen head", args.head_epochs, 0.003), ("Fine-tune layer4", args.fine_epochs, 0.0003)]
    else:
        stages = [("From scratch", args.head_epochs + args.fine_epochs, 0.001)]

    history, best_loss = [], float("inf")
    for stage, epochs, lr in stages:
        if stage == "Fine-tune layer4":
            model.load_state_dict(best_state)
            model.layer4.requires_grad_(True)
            for layer in model.layer4.modules():
                if isinstance(layer, nn.BatchNorm2d):
                    layer.requires_grad_(False)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
        print(f"{MODEL_NAMES[model_name]} / {stage} / {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters", flush=True)
        for _ in range(epochs):
            train_loss, train_accuracy, _ = epoch_pass(model, loaders["train"], device, optimizer, model_name == "resnet")
            val_loss, val_accuracy, _ = epoch_pass(model, loaders["validation"], device)
            history.append(dict(epoch=len(history) + 1, stage=stage, train_loss=train_loss,
                                val_loss=val_loss, train_accuracy=train_accuracy, val_accuracy=val_accuracy))
            print(f"  {len(history):02d} | train {train_accuracy:.3f} ({train_loss:.3f})"
                  f" | validation {val_accuracy:.3f} ({val_loss:.3f})", flush=True)
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = len(history)
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    torch.save(best_state, OUT / f"{model_name}.pt")
    return model, history, best_epoch


def stress_images(images):
    y, x = np.mgrid[:8, :8]
    texture = (0.15 + 0.12 * np.sin(x * 1.6 + y * 1.2) + 0.13 * x / 7).astype(np.float32)
    return {
        "dim": [ImageEnhance.Brightness(image).enhance(0.40) for image in images],
        "textured": [Image.fromarray(np.rint(
            255 * (texture + (1 - texture) * np.asarray(image) / 255)
        ).astype(np.uint8)) for image in images],
    }


def plot_curves(results, head_epochs):
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.8), layout="constrained")
    for row, (name, result) in enumerate(results.items()):
        records = result["history"]
        epochs = [r["epoch"] for r in records]
        for col, metric in enumerate(("accuracy", "loss")):
            ax = axes[row, col]
            for split, color in [("train", "#0c766e"), ("val", "#bd5719")]:
                ax.plot(epochs, [r[f"{split}_{metric}"] for r in records], ".-",
                        color=color, label="Training" if split == "train" else "Validation")
            if name == "resnet":
                ax.axvline(head_epochs + 0.5, ls="--", color="#6a7385", label="Fine-tuning begins")
            ax.set(title=f"{MODEL_NAMES[name]} · {metric.title()}", xlabel="Epoch", xticks=epochs)
            ax.grid(alpha=0.22)
            ax.legend(fontsize=8)
    fig.savefig(OUT / "training_curves.png", dpi=170)
    plt.close(fig)


def plot_confusion(targets, scores):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.65), layout="constrained")
    for ax, (name, predictions) in zip(axes, scores.items()):
        ConfusionMatrixDisplay.from_predictions(targets, predictions.argmax(1), labels=list(range(10)),
                                                ax=ax, colorbar=False, cmap="Blues",
                                                text_kw={"fontsize": 7})
        ax.set_title(MODEL_NAMES[name])
    fig.savefig(OUT / "confusion_matrices.png", dpi=170)
    plt.close(fig)


def write_examples(images, labels, test, scores, variants, stress):
    sample_path = OUT / "samples"
    sample_path.mkdir(exist_ok=True)
    selected = [next(k for k, i in enumerate(test) if labels[i] == digit) for digit in range(10)]
    samples = []
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.5), layout="constrained")
    predictions = scores["resnet"].argmax(1)
    for ax, k in zip(axes.flat, selected):
        index = int(test[k])
        filename = f"digit-{labels[index]}-{index}.png"
        images[index].save(sample_path / filename)
        samples.append({"file": filename, "index": index, "actual": int(labels[index])})
        ax.imshow(images[index], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(f"True {labels[index]} · Predicted {predictions[k]}\nScore {scores['resnet'][k].max():.1%}", fontsize=9)
        ax.axis("off")
    fig.savefig(OUT / "sample_predictions.png", dpi=160)
    plt.close(fig)
    (OUT / "samples.json").write_text(json.dumps(samples, indent=2) + "\n")

    # Show the most confident errors from the clean test and both stress conditions.
    cases = []
    for condition, probabilities in [("clean", scores["resnet"]), *stress["resnet"].items()]:
        errors = np.flatnonzero(probabilities.argmax(1) != labels[test])
        rank = errors[np.argsort(-probabilities[errors].max(axis=1))]
        cases.extend((condition, int(k), probabilities[k]) for k in rank[:3])
    fig, axes = plt.subplots(3, 3, figsize=(8, 6.3), layout="constrained")
    for ax in axes.flat:
        ax.axis("off")
    for row, condition in enumerate(("clean", "dim", "textured")):
        matches = [(k, p) for name, k, p in cases if name == condition]
        if not matches:
            axes[row, 1].text(0.5, 0.5, f"{condition.title()}: no errors", ha="center")
        for ax, (k, probabilities) in zip(axes[row], matches):
            i = int(test[k])
            image = images[i] if condition == "clean" else variants[condition][i]
            ax.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.set_title(f"{condition.title()} · #{i} · {labels[i]} → {probabilities.argmax()}\n"
                         f"Score {probabilities.max():.1%}", fontsize=9)
    fig.savefig(OUT / "error_examples.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, help="Local ImageNet ResNet18 safetensors file")
    parser.add_argument("--head-epochs", type=int, default=4)
    parser.add_argument("--fine-epochs", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.head_epochs, args.fine_epochs, args.threads) < 1:
        parser.error("Epoch counts and thread count must be positive.")

    if args.weights is None:
        from huggingface_hub import hf_hub_download

        args.weights = Path(hf_hub_download(WEIGHT_REPO, "model.safetensors", revision=WEIGHT_REVISION))
    weights_sha = hashlib.sha256(args.weights.read_bytes()).hexdigest()
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    OUT.mkdir(exist_ok=True)
    pixels, labels = load_digits(return_X_y=True)
    pixels = pixels.reshape(-1, 8, 8)
    splits = split_indices(pixels, labels)
    images = [digit_image(p) for p in pixels]
    test = splits["test"]
    variants = stress_images(images)
    with (OUT / "splits.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("index", "label", "split"))
        writer.writerows((i, labels[i], split) for split, indices in splits.items() for i in indices)
    results, scores, stress = {}, {}, {}
    for name in MODEL_NAMES:
        loaders = {split: make_loader(images, labels, indices, name, split == "train")
                   for split, indices in splits.items()}
        model, history, best_epoch = train_model(name, loaders, args, device, args.weights)
        test_loss, _, scores[name] = epoch_pass(model, loaders["test"], device)
        stress[name] = {
            condition: epoch_pass(model, make_loader(varied, labels, test, name), device)[2]
            for condition, varied in variants.items()
        }
        results[name] = {
            "history": history,
            "selected_epoch": best_epoch,
            "parameters": sum(p.numel() for p in model.parameters()),
            "test_loss": test_loss,
            "classification_report": classification_report(labels[test], scores[name].argmax(1), labels=list(range(10)),
                                                           output_dict=True, zero_division=0),
            "stress_accuracy": {k: accuracy_score(labels[test], v.argmax(1)) for k, v in stress[name].items()},
        }
        print(f"  Held-out accuracy: {results[name]['classification_report']['accuracy']:.2%}", flush=True)
    for name, result in results.items():
        print(f"{MODEL_NAMES[name]} stress accuracy {result['stress_accuracy']}", flush=True)

    with (OUT / "test_predictions.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("index", "actual", "model", "condition", "predicted", "score"))
        for name in MODEL_NAMES:
            for condition, values in {"clean": scores[name], **stress[name]}.items():
                writer.writerows((i, labels[i], name, condition, p.argmax(), f"{p.max():.8f}") for i, p in zip(test, values))
    experiment = {
        "seed": SEED, "device": args.device, "threads": args.threads,
        "torch": str(torch.__version__), "sklearn": sklearn_version, "weight_repo": WEIGHT_REPO,
        "weight_revision": WEIGHT_REVISION, "weight_sha256": weights_sha,
        "head_epochs": args.head_epochs, "fine_epochs": args.fine_epochs,
        "split_counts": {key: np.bincount(labels[indices], minlength=len(CLASSES)).tolist() for key, indices in splits.items()},
        "models": results,
    }
    (OUT / "metrics.json").write_text(json.dumps(experiment, indent=2) + "\n")
    plot_curves(results, args.head_epochs)
    plot_confusion(labels[test], scores)
    write_examples(images, labels, test, scores, variants, stress)


if __name__ == "__main__":
    main()
