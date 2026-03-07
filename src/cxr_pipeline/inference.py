from __future__ import annotations

from io import BytesIO

import torch
from PIL import Image

from .transforms import get_eval_transforms

_EVAL_TRANSFORMS = get_eval_transforms()


def load_image(source: str) -> Image.Image:
    if source.startswith(("http://", "https://")):
        import requests

        response = requests.get(source, timeout=30)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    return Image.open(source).convert("RGB")


def predict_image(
    source: str, model, device: str | torch.device, threshold: float = 0.5
) -> dict[str, object]:
    model.eval()
    image = load_image(source)
    image_tensor = _EVAL_TRANSFORMS(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        prob = torch.sigmoid(logits).item()

    prediction = int(prob > threshold)
    return {
        "source": source,
        "p_abnormal": round(prob, 4),
        "prediction": prediction,
        "label_name": "abnormal" if prediction == 1 else "normal",
    }
