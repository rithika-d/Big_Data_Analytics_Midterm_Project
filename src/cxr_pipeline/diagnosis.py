from __future__ import annotations

from collections.abc import Callable

import torch
from PIL import Image

from .inference import load_image, predict_image


def build_prompt(p_abnormal: float) -> str:
    if p_abnormal > 0.8:
        return f"""
A binary chest X-ray classifier assigned this image an abnormal probability of {p_abnormal:.2f}.

Given the strong abnormal signal:

• Describe the dominant and most likely radiographic patterns visible.
• Explain which imaging features most strongly support abnormality.
• Provide 2–3 likely explanations.
• Indicate level of certainty.
• Avoid definitive diagnosis.

Focus on prominent findings rather than subtle ones.
"""
    if p_abnormal <= 0.7:
        return f"""
A binary chest X-ray classifier assigned this image an abnormal probability of {p_abnormal:.2f}, indicating a borderline abnormal signal.

• Carefully describe any subtle or equivocal imaging findings.
• Discuss alternative normal variants that could mimic abnormality.
• Emphasize uncertainty.
• Provide low-confidence differential considerations only if justified.

Avoid strong conclusions and clearly state limitations.
"""
    return f"""
A binary chest X-ray classifier assigned this image an abnormal probability of {p_abnormal:.2f}.

• Describe observable imaging features.
• Provide a short differential diagnosis list.
• Indicate level of certainty.
• Avoid definitive medical diagnosis.
• Emphasize that interpretation is preliminary and image-only.
"""


def diagnose_chest_xray(
    image_path: str,
    eva_model,
    device: torch.device,
    llm_generate_fn: Callable[[Image.Image, str], str],
    threshold: float = 0.5,
) -> dict[str, object]:
    result = predict_image(image_path, eva_model, device, threshold=threshold)

    output = {
        "source": image_path,
        "p_abnormal": result["p_abnormal"],
        "y_pred": int(result["p_abnormal"] > threshold),
        "reasoning": None,
    }

    if output["p_abnormal"] <= threshold:
        return output

    image = load_image(image_path)
    prompt = build_prompt(float(output["p_abnormal"]))
    output["reasoning"] = llm_generate_fn(image, prompt)
    return output
