from __future__ import annotations

from collections.abc import Callable

import torch
from PIL import Image


def load_llama_model(
    model_name: str = "0llheaven/Llama-3.2-11B-Vision-Radiology-mini",
    load_in_4bit: bool = True,
):
    from unsloth import FastVisionModel

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name,
        load_in_4bit=load_in_4bit,
        use_gradient_checkpointing="unsloth",
    )
    FastVisionModel.for_inference(model)
    return model, tokenizer


def make_llama_generate_fn(
    model, tokenizer, max_new_tokens: int = 128
) -> Callable[[Image.Image, str], str]:
    def generate(image: Image.Image, prompt: str) -> str:
        image_copy = image.copy()
        image_copy.thumbnail((448, 448), Image.Resampling.LANCZOS)
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
            }
        ]
        text_prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = tokenizer(
            text=text_prompt,
            images=image_copy,
            add_special_tokens=False,
            return_tensors="pt",
        )
        device = next(model.parameters()).device
        inputs = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in inputs.items()
        }

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                use_cache=False,
            )

        return tokenizer.decode(output[0], skip_special_tokens=True).strip()

    return generate
