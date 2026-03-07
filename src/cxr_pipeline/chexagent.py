from __future__ import annotations

import os
import tempfile
from collections.abc import Callable

import torch
from PIL import Image


def load_chexagent(
    model_name: str = "StanfordAIMI/CheXagent-2-3b-srrg-findings",
    device: str | torch.device = "cuda",
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = torch.bfloat16 if str(device) != "cpu" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=device,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer


def make_chexagent_generate_fn(
    model, tokenizer, device: str | torch.device
) -> Callable[[Image.Image, str], str]:
    def generate(image: Image.Image, prompt: str) -> str:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_path = tmp_file.name

        try:
            image_to_save = image if image.mode == "RGB" else image.convert("RGB")
            image_to_save.save(tmp_path)
            query = tokenizer.from_list_format([{"image": tmp_path}, {"text": prompt}])
            conversation = [
                {"from": "system", "value": "You are a helpful assistant."},
                {"from": "human", "value": query},
            ]
            input_ids = tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            output = model.generate(
                input_ids.to(device),
                do_sample=False,
                num_beams=1,
                temperature=1.0,
                top_p=1.0,
                use_cache=True,
                max_new_tokens=512,
            )[0]
            return tokenizer.decode(output[input_ids.size(1) : -1])
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    return generate
