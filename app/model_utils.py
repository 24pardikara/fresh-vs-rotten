"""Model definition + inference helpers for the Fresh-vs-Rotten app.

The MobileNetV3 head here is built *identically* to the training notebook so the
saved `state_dict` loads cleanly. Keep the two in sync if you change the head.
"""
from __future__ import annotations
import json
import os
import time

import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
from torchvision import models

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
WEIGHTS_PATH = os.path.join(MODELS_DIR, "mobilenetv3_fresh_rotten.pt")
LABELS_PATH = os.path.join(MODELS_DIR, "labels.json")

# Fallback label metadata if labels.json is missing (matches the notebook).
_DEFAULT_LABELS = {
    "classes": ["freshapples", "freshbanana", "freshoranges",
                "rottenapples", "rottenbanana", "rottenoranges"],
    "pretty": ["Fresh Apple", "Fresh Banana", "Fresh Orange",
               "Rotten Apple", "Rotten Banana", "Rotten Orange"],
    "arch": "mobilenet_v3_large",
    "img_size": 224,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}


def load_labels() -> dict:
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH) as f:
            return json.load(f)
    return _DEFAULT_LABELS


def build_mobilenetv3(num_classes: int, arch: str = "mobilenet_v3_large") -> nn.Module:
    """Build the MobileNetV3 variant used during training (no ImageNet weights
    needed here; we load the fine-tuned state_dict instead). The arch is read
    from labels.json so local (small) and Colab (large) runs both load cleanly."""
    if arch == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
    else:
        model = models.mobilenet_v3_large(weights=None)
    in_f = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_f, num_classes)
    return model


def load_model():
    """Returns (model, labels). Raises FileNotFoundError if weights are missing."""
    labels = load_labels()
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(
            f"Model weights not found at {WEIGHTS_PATH}. "
            "Train the model and place mobilenetv3_fresh_rotten.pt in app/models/."
        )
    model = build_mobilenetv3(num_classes=len(labels["classes"]),
                              arch=labels.get("arch", "mobilenet_v3_large"))
    state = torch.load(WEIGHTS_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, labels


def build_transform(labels: dict):
    size = labels.get("img_size", 224)
    mean = labels.get("mean", _DEFAULT_LABELS["mean"])
    std = labels.get("std", _DEFAULT_LABELS["std"])
    return T.Compose([
        T.Resize((256, 256)),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


@torch.no_grad()
def predict(model, labels: dict, image: Image.Image):
    """Run inference on a PIL image.

    Returns: (pretty_label, confidence_float, probs_dict, latency_ms)
    """
    tf = build_transform(labels)
    x = tf(image.convert("RGB")).unsqueeze(0)
    t0 = time.perf_counter()
    logits = model(x)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    probs = torch.softmax(logits, dim=1).squeeze(0)
    idx = int(probs.argmax().item())
    pretty = labels["pretty"][idx]
    conf = float(probs[idx].item())
    prob_map = {labels["pretty"][i]: float(probs[i].item())
                for i in range(len(labels["pretty"]))}
    return pretty, conf, prob_map, latency_ms
