"""Model definitions and the shared training and inference image pipeline."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision.models import resnet18
from torchvision.transforms import functional as F

SIZE = 64
CLASSES = tuple(map(str, range(10)))
MODEL_NAMES = {"cnn": "Basic CNN", "resnet": "ResNet18"}
ROOT = Path(__file__).resolve().parent
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def preprocess(image: Image.Image, model_name: str) -> torch.Tensor:
    image = ImageOps.exif_transpose(image).convert("L").resize((SIZE, SIZE), Image.Resampling.BILINEAR)
    tensor = F.to_tensor(image)
    if model_name == "resnet":
        return F.normalize(tensor.expand(3, -1, -1), MEAN, STD)
    return tensor


def digit_image(pixels: np.ndarray) -> Image.Image:
    return Image.fromarray(np.rint(pixels * (255.0 / 16)).astype(np.uint8))


def build_model(model_name: str, imagenet_weights=None) -> nn.Module:
    if model_name == "cnn":
        return nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            nn.Linear(1024, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, len(CLASSES)),
        )
    if model_name != "resnet":
        raise ValueError(f"Unknown model: {model_name}")
    model = resnet18()
    if imagenet_weights:
        from safetensors.torch import load_file

        model.load_state_dict(load_file(imagenet_weights))
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    return model


def load_model(model_name: str) -> nn.Module:
    model = build_model(model_name)
    weights = torch.load(ROOT / "outputs" / f"{model_name}.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(weights)
    return model.eval()


@torch.inference_mode()
def predict(model: nn.Module, image: Image.Image, model_name: str) -> list[dict]:
    probabilities = model(preprocess(image, model_name).unsqueeze(0)).softmax(1).squeeze(0)
    return [
        {"digit": CLASSES[index], "score": float(probabilities[index])}
        for index in probabilities.argsort(descending=True).tolist()
    ]
