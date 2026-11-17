"""Optional ResNet50 binary classifier (1 = 1980s, 0 = non-1980s).

You train this in Google Colab and save it with:
    torch.save(model.state_dict(), "timelens_resnet50_1980s.pth")
Put the file in models/. If torch or the file is missing, the app still works;
it simply skips the classifier and says so in the response (no fake scores).
"""
import os
from pathlib import Path

MODEL_PATH = Path(os.environ.get(
    "CLASSIFIER_PATH",
    Path(__file__).resolve().parent.parent / "models" / "timelens_resnet50_1980s.pth"))

_model = None
_tried = False
_torch = None
_transform = None


def _load():
    global _model, _tried, _torch, _transform
    if _tried:
        return _model
    _tried = True
    if not MODEL_PATH.exists():
        return None
    try:
        import torch
        import torch.nn as nn
        from torchvision import models, transforms
    except ImportError:
        return None
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 2)   # must match your training code
    m.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    m.eval()
    _torch = torch
    _transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    _model = m
    return _model


def is_available() -> bool:
    return _load() is not None


def predict_1980s_probability(image_path):
    """Return P(1980s) in [0,1], or None if the classifier is unavailable."""
    m = _load()
    if m is None:
        return None
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    x = _transform(img).unsqueeze(0)
    with _torch.no_grad():
        probs = _torch.softmax(m(x), dim=1)[0]
    return round(float(probs[1]), 4)
